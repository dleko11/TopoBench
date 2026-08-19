"""Cluster-GCN dataloading and streaming pipeline for topological deep learning."""

import glob
import hashlib
import math
import os
import os.path as osp
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import filelock
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from topobench.utils.phase_tracking import track_phase


class _HandleAdapter:
    """Minimal dataset-like adapter for the CSR collator.

    Parameters
    ----------
    handle : dict
        Metadata handle from the global partitioner.
    """

    def __init__(self, handle: dict[str, object]) -> None:
        self.processed_dir = handle["processed_dir"]
        self.num_parts = int(handle["num_parts"])
        self.sparse_format = str(handle["sparse_format"])


class _PartIdListDataset(Dataset):
    """Dataset that yields part IDs from a provided list.

    Parameters
    ----------
    part_ids : Sequence of int
        Cluster part IDs.
    """

    def __init__(self, part_ids: Sequence[int]) -> None:
        self._parts = np.asarray(part_ids, dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of elements in the dataset.

        Returns
        -------
        int
            The number of parts.
        """
        return self._parts.shape[0]

    def __getitem__(self, idx: int) -> int:
        """Return the part ID at index idx.

        Parameters
        ----------
        idx : int
            The index to access.

        Returns
        -------
        int
            The part ID.
        """
        return int(self._parts[idx])


def _make_hash(o: Any) -> int:
    """Generate a stable hash for cache keys.

    Parameters
    ----------
    o : Any
        The object to hash.

    Returns
    -------
    int
        Hash value restricted to uint32 range.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    # restrict into uint32 range for nicer filenames
    return int(sha1.hexdigest(), 16) % 4294967295


class _CachedBatchDataset(Dataset):
    """Dataset serving precomputed Data objects from disk.

    Parameters
    ----------
    files : list of str
        Paths to precomputed PyG Data files.
    """

    def __init__(self, files: list[str]):
        self.files = list(files)

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns
        -------
        int
            The number of files.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        """Load and return the Data object at index idx.

        Parameters
        ----------
        idx : int
            The index to access.

        Returns
        -------
        Data
            The loaded PyG Data object.
        """
        data = torch.load(
            self.files[idx], map_location="cpu", weights_only=False
        )
        return _ensure_cluster_batch_metadata(data)


def _identity_data_collate(batch_list: list[Data]) -> Data:
    """Collate a single-batch list by returning the first element.

    Parameters
    ----------
    batch_list : list of Data
        List containing a single Data object.

    Returns
    -------
    Data
        The first Data object in the list.
    """
    return batch_list[0]


def _ensure_cluster_batch_metadata(data: Data) -> Data:
    """Ensure cluster batches expose standard lifted batch metadata.

    Parameters
    ----------
    data : Data
        Cluster batch data object.

    Returns
    -------
    Data
        Data object with ``batch_*`` and ``cell_statistics`` fields.
    """
    if data.get("x") is not None and data.get("x_0") is None:
        data.x_0 = data.x

    if data.get("batch_0") is None:
        if data.get("batch") is not None:
            data.batch_0 = data.batch
        else:
            data.batch_0 = torch.zeros(data.num_nodes, dtype=torch.long)

    shape = data.get("shape")
    if shape is not None and data.get("cell_statistics") is None:
        cell_statistics = torch.as_tensor(shape, dtype=torch.long)
        if cell_statistics.dim() == 1:
            cell_statistics = cell_statistics.unsqueeze(0)
        data.cell_statistics = cell_statistics

    for key in list(data.keys()):
        if key.startswith("x_") and key not in ("x", "x_0"):
            if key == "x_hyperedges":
                cell_dim = "hyperedges"
            else:
                try:
                    cell_dim = int(key.split("_")[1])
                except Exception:
                    continue

            batch_key = f"batch_{cell_dim}"
            if data.get(batch_key) is None:
                num_cells = getattr(data, key).shape[0]
                setattr(
                    data,
                    batch_key,
                    torch.zeros(num_cells, dtype=torch.long),
                )

    return data


# Collator: stream CSR blocks + masks
class BlockCSRBatchCollator:
    """Collate Cluster-GCN mini-batches from CSR memmaps.

    Streams cluster blocks from disk (CSR structure, optional features,
    labels and edge attributes) and builds a :class:`Data` batch with
    node-level supervision masks and global node IDs.

    Parameters
    ----------
    ds_like : _HandleAdapter
        Adapter providing access to cluster metadata and memmaps.
    device : torch.device or None, optional
        Device to move the batch to. If ``None``, stays on CPU.
    with_edge_attr : bool, optional
        If True, reads and includes edge attributes. Default is False.
    reconstruct_cross_cluster_edges : bool, optional
        If True, retain edges between sampled clusters. If False, retain only
        edges whose endpoints belong to the same cluster. Default is True.
    active_split : {"train", "val", "test"}, optional
        Active split whose supervision mask is used. Default is "train".
    post_batch_transform : callable or None, optional
        Optional transform applied to the assembled batch.
    """

    def __init__(
        self,
        ds_like: _HandleAdapter,
        *,
        device: torch.device | None = None,
        with_edge_attr: bool = False,
        reconstruct_cross_cluster_edges: bool = True,
        active_split: str = "train",  # "train" | "val" | "test"
        post_batch_transform: Callable[..., Any] | None = None,
    ) -> None:
        self.ds = ds_like
        self.device = device
        self.with_edge_attr = with_edge_attr
        self.reconstruct_cross_cluster_edges = bool(
            reconstruct_cross_cluster_edges
        )
        self.active_split = str(active_split).lower()
        assert self.active_split in ("train", "val", "test")
        self.post_batch_transform = post_batch_transform

        mm_dir = osp.join(self.ds.processed_dir, "perm_memmap")
        # Structural memmaps:
        self.partptr = np.load(osp.join(mm_dir, "partptr.npy"), mmap_mode="r")
        self.indptr = np.load(osp.join(mm_dir, "indptr.npy"), mmap_mode="r")
        self.indices = np.load(osp.join(mm_dir, "indices.npy"), mmap_mode="r")

        # Optional arrays:
        self.X = None
        self.Y = None
        self.EA = None
        x_path = osp.join(mm_dir, "X_perm.npy")
        y_path = osp.join(mm_dir, "y_perm.npy")
        ea_path = osp.join(mm_dir, "edge_attr_perm.npy")
        if osp.exists(x_path):
            self.X = np.load(x_path, mmap_mode="r")
        if osp.exists(y_path):
            self.Y = np.load(y_path, mmap_mode="r")
        if with_edge_attr and osp.exists(ea_path):
            self.EA = np.load(ea_path, mmap_mode="r")

        # Split masks (permuted)
        m_train = osp.join(mm_dir, "train_mask_perm.npy")
        m_val = osp.join(mm_dir, "val_mask_perm.npy")
        m_test = osp.join(mm_dir, "test_mask_perm.npy")
        if not (
            osp.exists(m_train) and osp.exists(m_val) and osp.exists(m_test)
        ):
            raise FileNotFoundError(
                "Permuted split masks not found in memmap dir."
            )
        self.train_mask_perm = np.load(m_train, mmap_mode="r")
        self.val_mask_perm = np.load(m_val, mmap_mode="r")
        self.test_mask_perm = np.load(m_test, mmap_mode="r")

        assert self.ds.sparse_format == "csr", (
            f"Expected CSR, got {self.ds.sparse_format}"
        )

        self._transform_calls = 0

    def _active_mask_array(self) -> np.ndarray:
        """Choose and return the active split mask array.

        Returns
        -------
        np.ndarray
            The permuted split mask array corresponding to active_split.
        """
        if self.active_split == "train":
            return self.train_mask_perm
        if self.active_split == "val":
            return self.val_mask_perm
        return self.test_mask_perm

    def __call__(self, parts: list[int]) -> Data:
        """Build a union batch from a list of cluster IDs.

        For the given cluster IDs, collects their CSR rows, node
        features, labels and (optionally) edge attributes, then returns
        a single :class:`Data` object.

        Parameters
        ----------
        parts : list of int
            Cluster IDs to merge into a mini-batch (length == ``q``).

        Returns
        -------
        Data
            Batched graph with fields such as ``edge_index``, ``x``,
            ``y``, ``edge_attr``, ``supervised_mask`` and ``global_nid``.
        """
        # ranges for selected clusters (sorted for monotonic slices)
        parts = np.asarray(parts, dtype=np.int64)
        starts = self.partptr[parts]
        ends = self.partptr[parts + 1]
        order = np.argsort(starts)
        starts, ends = starts[order], ends[order]

        # gather node features/labels and build global_nid list
        offsets = np.zeros(len(starts), dtype=np.int64)
        total_nodes = 0
        xs, ys = [], []
        global_ids = []
        for i, (s, e) in enumerate(zip(starts, ends, strict=False)):
            offsets[i] = total_nodes
            # append features
            if self.X is not None:
                xs.append(torch.from_numpy(self.X[s:e]))
            # append labels
            if self.Y is not None:
                ys.append(torch.from_numpy(self.Y[s:e]))
            # append global permuted ids for these rows
            if e > s:
                global_ids.append(np.arange(s, e, dtype=np.int64))
            total_nodes += e - s

        x = torch.cat(xs, dim=0) if xs else None
        y = torch.cat(ys, dim=0) if ys else None
        global_ids = (
            np.concatenate(global_ids, axis=0)
            if len(global_ids)
            else np.empty((0,), dtype=np.int64)
        )

        # stream CSR rows for each [s:e) -> make row/col (global ids)
        row_chunks, col_chunks, source_part_chunks = [], [], []
        ea_chunks = [] if self.EA is not None else None

        for part_index, (s, e, off) in enumerate(
            zip(starts, ends, offsets, strict=False)
        ):
            rowptr = self.indptr[s : e + 1]  # shape (e-s+1,)
            deg = rowptr[1:] - rowptr[:-1]  # per-row degrees
            beg, fin = (
                int(rowptr[0]),
                int(rowptr[-1]),
            )  # contiguous span in indices
            cols = torch.from_numpy(
                self.indices[beg:fin].astype(np.int64, copy=False)
            )
            rows = torch.arange(e - s, dtype=torch.int64).repeat_interleave(
                torch.from_numpy(deg.astype(np.int64))
            ) + int(off)
            row_chunks.append(rows)
            col_chunks.append(cols)
            source_part_chunks.append(
                torch.full((cols.numel(),), part_index, dtype=torch.int64)
            )

            if ea_chunks is not None:
                ea_chunks.append(torch.from_numpy(self.EA[beg:fin]))

        row = (
            torch.cat(row_chunks, dim=0)
            if row_chunks
            else torch.empty(0, dtype=torch.int64)
        )
        col = (
            torch.cat(col_chunks, dim=0)
            if col_chunks
            else torch.empty(0, dtype=torch.int64)
        )
        source_part = (
            torch.cat(source_part_chunks, dim=0)
            if source_part_chunks
            else torch.empty(0, dtype=torch.int64)
        )
        edge_attr = torch.cat(ea_chunks, dim=0) if ea_chunks else None

        # keep only edges whose dst is inside the union of selected ranges
        starts_t = torch.from_numpy(starts)
        ends_t = torch.from_numpy(ends)
        offsets_t = torch.from_numpy(offsets)

        idx = torch.bucketize(col, starts_t, right=True) - 1
        valid = (idx >= 0) & (col < ends_t.gather(0, idx.clamp_min(0)))
        if not self.reconstruct_cross_cluster_edges:
            valid &= idx == source_part

        row = row[valid]
        col = col[valid]
        idx = idx[valid]
        if edge_attr is not None:
            edge_attr = edge_attr[valid]

        # global->local column ids: col_local = col - starts[idx] + offsets[idx]
        col_local = col - starts_t.gather(0, idx) + offsets_t.gather(0, idx)
        edge_index = torch.stack([row, col_local], dim=0)

        data = Data(edge_index=edge_index)
        if x is not None:
            data.x = x
        if y is not None:
            data.y = y
        if edge_attr is not None:
            data.edge_attr = edge_attr
        data.num_nodes = int(total_nodes)

        # ---- split-specific masks & ids ----
        active_mask = self._active_mask_array()
        supervised_mask = torch.from_numpy(active_mask[global_ids]).to(
            torch.bool
        )
        global_nid = torch.from_numpy(global_ids.astype(np.int64, copy=False))

        # Apply transforms on the full batch (LIFTING goes here)
        if self.post_batch_transform is not None:
            self._transform_calls += 1
            if self.active_split == "val":
                print(
                    f"[VAL] post_batch_transform call #{self._transform_calls}"
                )
            data = self.post_batch_transform(data)

        data.supervised_mask = supervised_mask
        data.global_nid = global_nid

        # Backwards-compatible attributes expected by existing code:
        if self.active_split == "train":
            data.train_mask = supervised_mask
        elif self.active_split == "val":
            data.val_mask = supervised_mask
        elif self.active_split == "test":
            data.test_mask = supervised_mask

        data = _ensure_cluster_batch_metadata(data)

        if self.device is not None:
            data = data.to(self.device)

        return data


def _process_and_save_batch(task):
    """Execute a single batch task and save it to disk.

    Parameters
    ----------
    task : tuple
        Tuple containing index, parts, path, handle, edge options, split, and
        transform configuration.

    Returns
    -------
    tuple
        Tuple of (index, final_path, duration).
    """
    (
        i,
        parts,
        final_path,
        handle,
        with_edge_attr,
        reconstruct_cross_cluster_edges,
        split,
        transform_config,
    ) = task
    import os
    import time

    import torch

    start_time = time.time()

    # Reconstruct transform from config to avoid pickling issues
    post_batch_transform = None
    if transform_config is not None:
        from omegaconf import OmegaConf

        from topobench.data.utils import build_cluster_transform

        cfg = OmegaConf.create(transform_config)
        post_batch_transform = build_cluster_transform(cfg)

    ds_adapter = _HandleAdapter(handle)

    collate = BlockCSRBatchCollator(
        ds_adapter,
        device=None,
        with_edge_attr=with_edge_attr,
        reconstruct_cross_cluster_edges=reconstruct_cross_cluster_edges,
        active_split=split,
        post_batch_transform=post_batch_transform,
    )

    data = collate(parts)
    data = data.cpu()

    tmp_path = final_path + f".tmp.{os.getpid()}"
    torch.save(data, tmp_path)
    os.replace(tmp_path, final_path)

    return i, final_path, time.time() - start_time


# DataModule-like wrapper
class ClusterGCNDataModule(LightningDataModule):
    """Streaming DataModule for a single global Cluster-GCN partition.

    Uses one shared global partition and memmap bundle; train, validation
    and test loaders differ only in which cluster parts they cover and
    which supervision mask is active.

    Parameters
    ----------
    data_handle : dict
        Handle dictionary describing dataset paths and metadata.
    q : int, optional
        Number of clusters per mini-batch. Default is 10.
    q_val : int, optional
        Number of clusters per mini-batch for validation. Default is None.
    q_test : int, optional
        Number of clusters per mini-batch for testing. Default is None.
    val_batches : int, optional
        Target number of validation batches. Default is 5.
    test_batches : int, optional
        Target number of test batches. Default is None.
    num_workers : int, optional
        Number of worker processes for the dataloaders. Default is 0.
    pin_memory : bool, optional
        If True, pin memory in dataloaders. Default is False.
    with_edge_attr : bool, optional
        If True, batches include edge attributes. Default is False.
    reconstruct_cross_cluster_edges : bool, optional
        If True, retain edges between clusters sampled into the same batch.
        Default is True.
    train_shuffle : bool, optional
        If True, reshuffle cluster grouping each training epoch. If False,
        keep deterministic cluster groups. Default is True.
    eval_cover_strategy : str, optional
        Strategy for evaluation coverage. Default is "all_parts".
    seed : int, optional
        Random seed for part shuffling. Default is 42.
    device : torch.device or None, optional
        Device to move batches to. Default is None.
    persistent_workers : bool or None, optional
        If True, use persistent workers in dataloaders. Default is None.
    transform_config : dict, optional
        Optional transform configuration dictionary. Default is None.
    post_batch_transform : callable or None, optional
        Optional transform applied to each batch after collation. Default is None.
    cache_num_workers : int, optional
        Number of worker processes for cache generation. Default is None.
    cache_val : bool, optional
        If True, validation batches are precomputed and cached. Default is True.
    val_cache_dir : str, optional
        Custom validation cache directory path. Default is None.
    val_cache_fingerprint : str or int, optional
        Fingerprint to differentiate cache runs. Default is None.
    """

    def __init__(
        self,
        *,
        data_handle: dict[str, object],
        q: int = 10,
        q_val: int | None = None,
        q_test: int | None = None,
        val_batches: int | None = 5,
        test_batches: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        with_edge_attr: bool = False,
        reconstruct_cross_cluster_edges: bool = True,
        train_shuffle: bool = True,
        eval_cover_strategy: str = "all_parts",
        seed: int = 42,
        device: torch.device | None = None,
        persistent_workers: bool | None = None,
        transform_config: dict | None = None,
        post_batch_transform: Callable[..., Any] | None = None,
        cache_num_workers: int | None = None,
        cache_val: bool = True,
        val_cache_dir: str | None = None,
        val_cache_fingerprint: int | str | None = None,
    ) -> None:
        super().__init__()

        self.handle = data_handle
        self._num_parts = int(self.handle.get("num_parts"))

        self.q = int(q)

        if q_val is not None:
            self.q_val = int(q_val)
        else:
            vb = int(val_batches) if val_batches is not None else None
            if vb is None:
                self.q_val = self.q
            else:
                self.q_val = max(q, math.ceil(self._num_parts / vb))

        if q_test is not None:
            self.q_test = int(q_test)
        else:
            tb = int(test_batches) if test_batches is not None else None
            if tb is None:
                self.q_test = self._num_parts
            else:
                self.q_test = max(q, math.ceil(self._num_parts / tb))

        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.with_edge_attr = bool(with_edge_attr)
        self.reconstruct_cross_cluster_edges = bool(
            reconstruct_cross_cluster_edges
        )
        self.train_shuffle = bool(train_shuffle)
        self.eval_cover_strategy = str(eval_cover_strategy)
        self.seed = int(seed)
        self.device = device
        self.persistent_workers = (
            bool(persistent_workers)
            if persistent_workers is not None
            else (self.num_workers > 0)
        )

        self.ds_adapter = _HandleAdapter(self.handle)
        self._paths = self.handle.get("paths", {})
        self.transform_config = transform_config
        self.cache_num_workers = (
            int(cache_num_workers) if cache_num_workers is not None else None
        )

        if post_batch_transform is not None:
            self.post_batch_transform = post_batch_transform
        elif transform_config is not None:
            from omegaconf import OmegaConf

            from topobench.data.utils import build_cluster_transform

            self.post_batch_transform = build_cluster_transform(
                OmegaConf.create(transform_config)
            )
        else:
            self.post_batch_transform = None

        # Preload part-lists for splits if available
        self._parts_with = {}
        for split in ("train", "val", "test"):
            key = f"parts_with_{split}"
            path = self._paths.get(key, None)
            if path and osp.exists(path):
                self._parts_with[split] = np.load(path)
            else:
                self._parts_with[split] = None

        self.cache_val = bool(cache_val)
        self.val_cache_dir = val_cache_dir
        self.val_cache_fingerprint = val_cache_fingerprint
        self._val_cache_files: list[str] | None = None

    def _part_ids_for_split(self, split: str) -> Iterable[int]:
        """Return cluster IDs to iterate for a given split.

        Parameters
        ----------
        split : str
            The dataset split ("train", "val", "test").

        Returns
        -------
        Iterable of int
            Cluster part IDs for the split.
        """
        split = split.lower()

        key = None
        if split == "train":
            key = "train"
        elif split == "val":
            key = "val"
        elif split == "test":
            key = "test"

        if key is not None:
            arr = self._parts_with.get(key, None)
            if arr is not None and len(arr) > 0:
                return arr.astype(np.int64)

        # Fallback: if parts_with_* is missing, use all parts.
        return np.arange(self.ds_adapter.num_parts, dtype=np.int64)

    def _part_ids_for_coverage(
        self,
        *,
        split: str,
        cover_parts: str,
    ) -> Iterable[int]:
        """Return part IDs for an inference coverage mode.

        Parameters
        ----------
        split : str
            Dataset split whose split-aware part list may be used.
        cover_parts : {"split", "all"}
            ``"split"`` uses parts containing supervised nodes for the split.
            ``"all"`` uses every partition part.

        Returns
        -------
        Iterable of int
            Cluster part IDs to iterate.
        """
        cover_parts = str(cover_parts).lower()
        if cover_parts == "split":
            return self._part_ids_for_split(split)
        if cover_parts == "all":
            return np.arange(self.ds_adapter.num_parts, dtype=np.int64)
        raise ValueError(
            "cover_parts must be either 'split' or 'all', "
            f"got {cover_parts!r}."
        )

    def _q_for_split(self, split: str) -> int:
        """Return the batch size (number of clusters) for a given split.

        Parameters
        ----------
        split : str
            The dataset split ("train", "val", "test").

        Returns
        -------
        int
            Batch size for the given split.
        """
        split = split.lower()
        if split == "val":
            return self.q_val
        if split == "test":
            return self.q_test
        return self.q

    @property
    def num_parts(self) -> int:
        """Return the number of partition parts.

        Returns
        -------
        int
            Number of partition parts in the global partition.
        """
        return self._num_parts

    def _build_loader(
        self,
        *,
        split: str,
        shuffle: bool,
        q: int | None = None,
        seed: int | None = None,
        cover_parts: str = "split",
    ) -> DataLoader:
        """Build and return a DataLoader for a given split.

        Parameters
        ----------
        split : str
            The dataset split ("train", "val", "test").
        shuffle : bool
            Whether to shuffle the dataset parts.
        q : int or None, optional
            Number of clusters per mini-batch. Defaults to the split setting.
        seed : int or None, optional
            Seed used when ``shuffle`` is True. Defaults to the datamodule seed.
        cover_parts : {"split", "all"}, optional
            Part coverage mode. Default is ``"split"``.

        Returns
        -------
        DataLoader
            DataLoader for the split.
        """
        part_ids = self._part_ids_for_coverage(
            split=split,
            cover_parts=cover_parts,
        )
        part_ds = _PartIdListDataset(part_ids)
        batch_size = self._q_for_split(split) if q is None else int(q)
        if batch_size <= 0:
            raise ValueError(f"q must be positive, got {batch_size}.")

        collate = BlockCSRBatchCollator(
            self.ds_adapter,
            device=self.device,
            with_edge_attr=self.with_edge_attr,
            reconstruct_cross_cluster_edges=(
                self.reconstruct_cross_cluster_edges
            ),
            active_split=split,
            post_batch_transform=self.post_batch_transform,
        )

        g = torch.Generator()
        g.manual_seed(self.seed if seed is None else int(seed))

        return DataLoader(
            part_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate,
            generator=g if shuffle else None,
            drop_last=False,
        )

    def inference_dataloader(
        self,
        *,
        split: str = "test",
        q: int | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        cover_parts: str = "split",
    ) -> DataLoader:
        """Build an inference dataloader with explicit grouping controls.

        Parameters
        ----------
        split : str, optional
            Dataset split to evaluate. Default is ``"test"``.
        q : int or None, optional
            Number of clusters per mini-batch. Defaults to the split setting.
        shuffle : bool, optional
            Whether to shuffle part order. Default is False.
        seed : int or None, optional
            Seed used when shuffling. Defaults to the datamodule seed.
        cover_parts : {"split", "all"}, optional
            Whether to cover split-supervised parts or all parts. Default is
            ``"split"``.

        Returns
        -------
        DataLoader
            Inference dataloader.
        """
        return self._build_loader(
            split=split,
            shuffle=shuffle,
            q=q,
            seed=seed,
            cover_parts=cover_parts,
        )

    def setup(self, stage: str | None = None) -> None:
        """Precompute and cache validation batches (after lifting) once.

        Parameters
        ----------
        stage : str or None, optional
            The stage parameter ("fit", "validate", "test"). Default is None.
        """
        if stage not in (None, "fit", "validate"):
            return
        if not self.cache_val:
            return

        # Decide base cache folder
        base_dir = self.val_cache_dir
        if base_dir is None:
            base_dir = osp.join(self.ds_adapter.processed_dir, "val_cache")
        os.makedirs(base_dir, exist_ok=True)

        # Decide cache identifier (fingerprint recommended)
        if self.val_cache_fingerprint is not None:
            cache_id = str(self.val_cache_fingerprint)
        else:
            cache_id = str(
                _make_hash(
                    {
                        "q_val": self.q_val,
                        "with_edge_attr": int(self.with_edge_attr),
                        "reconstruct_cross_cluster_edges": int(
                            self.reconstruct_cross_cluster_edges
                        ),
                        "seed": self.seed,
                        "eval_cover_strategy": self.eval_cover_strategy,
                    }
                )
            )

        cache_dir = osp.join(base_dir, f"val_{cache_id}")
        os.makedirs(cache_dir, exist_ok=True)

        existing = sorted(glob.glob(osp.join(cache_dir, "batch_*.pt")))
        complete_marker = osp.join(cache_dir, "_COMPLETE")
        if len(existing) > 0 and osp.exists(complete_marker):
            self._val_cache_files = existing
            return

        lock_path = osp.join(base_dir, f"val_{cache_id}.lock")

        with filelock.FileLock(lock_path, timeout=-1):
            existing = sorted(glob.glob(osp.join(cache_dir, "batch_*.pt")))
            if len(existing) > 0 and osp.exists(complete_marker):
                self._val_cache_files = existing
                return

            for f in existing:
                os.remove(f)

            part_ids = np.asarray(
                list(self._part_ids_for_split("val")), dtype=np.int64
            )

            batches = [
                part_ids[i : i + self.q_val].tolist()
                for i in range(0, len(part_ids), self.q_val)
            ]

            cache_files: list[str] = []

            num_workers = self.cache_num_workers
            if num_workers is None:
                num_workers = self.num_workers

            cache_tracking_extra = {
                "tracking/val_cache_batches": len(batches),
                "tracking/val_cache_workers": int(num_workers),
            }
            with track_phase("val_cache_build", extra=cache_tracking_extra):
                if num_workers > 1:
                    import logging
                    from concurrent.futures import (
                        ProcessPoolExecutor,
                        as_completed,
                    )

                    logging.info(
                        f"[VAL] Building cache with {num_workers} workers: {len(batches)} batches"
                    )

                    tasks = []
                    for i, parts in enumerate(batches):
                        final_path = osp.join(cache_dir, f"batch_{i:04d}.pt")
                        tasks.append(
                            (
                                i,
                                parts,
                                final_path,
                                self.handle,
                                self.with_edge_attr,
                                self.reconstruct_cross_cluster_edges,
                                "val",
                                self.transform_config,
                            )
                        )

                    cache_files = [None] * len(tasks)
                    with ProcessPoolExecutor(
                        max_workers=num_workers
                    ) as executor:
                        futures = {
                            executor.submit(
                                _process_and_save_batch,
                                task,
                            ): task[0]
                            for task in tasks
                        }
                        for future in as_completed(futures):
                            idx = futures[future]
                            _, final_path, duration = future.result()
                            cache_files[idx] = final_path

                else:
                    collate = BlockCSRBatchCollator(
                        self.ds_adapter,
                        device=None,
                        with_edge_attr=self.with_edge_attr,
                        reconstruct_cross_cluster_edges=(
                            self.reconstruct_cross_cluster_edges
                        ),
                        active_split="val",
                        post_batch_transform=self.post_batch_transform,
                    )
                    import logging

                    logging.info(
                        f"[VAL] Building cache with serial fallback: {len(batches)} batches"
                    )
                    for i, parts in enumerate(batches):
                        data = collate(parts)
                        data = data.cpu()
                        tmp_path = osp.join(cache_dir, f"batch_{i:04d}.pt.tmp")
                        final_path = osp.join(cache_dir, f"batch_{i:04d}.pt")
                        torch.save(data, tmp_path)
                        os.replace(tmp_path, final_path)
                        cache_files.append(final_path)

                with open(complete_marker, "w") as f:
                    f.write("done")

        self._val_cache_files = cache_files

    def train_dataloader(self) -> DataLoader:
        """Return dataloader for the training split.

        Returns
        -------
        DataLoader
            Training dataloader.
        """
        return self._build_loader(
            split="train",
            shuffle=self.train_shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        """Return dataloader for the validation split.

        If caching is enabled and present, returns cached lifted batches.

        Returns
        -------
        DataLoader
            Validation dataloader.
        """
        if self.cache_val and self._val_cache_files is not None:
            ds = _CachedBatchDataset(self._val_cache_files)

            def _cached_collate(batch_list: list[Data]) -> Data:
                data = batch_list[0]  # batch_size=1
                if self.device is not None:
                    data = data.to(self.device, non_blocking=True)
                return data

            return DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=self.pin_memory,
                collate_fn=_cached_collate,
                persistent_workers=False,
            )

        return self._build_loader(split="val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return dataloader for the test split.

        Returns
        -------
        DataLoader
            Test dataloader.
        """
        return self._build_loader(split="test", shuffle=False)
