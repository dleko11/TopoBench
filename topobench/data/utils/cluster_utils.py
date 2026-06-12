"""Utilities for building and storing Cluster-GCN partition data."""

import os
import os.path as osp
from collections.abc import Callable
from typing import Any

import hydra
import numpy as np
import torch
import torch_geometric
from numpy.lib.format import open_memmap
from torch_geometric.data import Data, OnDiskDataset
from torch_geometric.loader import ClusterData


def build_cluster_transform(transforms_config) -> Callable | None:
    """Build a post-batch transform for clustered training.

    Parameters
    ----------
    transforms_config : dict or None
        Hydra-style configuration for transforms.

    Returns
    -------
    callable or None
        Composed transform or ``None`` if no transforms are defined.
    """
    if not transforms_config:
        return None

    if set(transforms_config.keys()) == {"liftings"}:
        transforms_config = transforms_config["liftings"]

    hydra.utils.instantiate(transforms_config)

    from topobench.transforms.data_transform import DataTransform

    transform_dict = {
        key: DataTransform(**value) for key, value in transforms_config.items()
    }

    if not transform_dict:
        return None

    if len(transform_dict) == 1:
        return next(iter(transform_dict.values()))

    return torch_geometric.transforms.Compose(list(transform_dict.values()))


def to_bool_mask(mask: torch.Tensor, N: int) -> torch.Tensor:
    """Convert an index or score tensor to a boolean mask.

    Parameters
    ----------
    mask : torch.Tensor
        Input mask, index tensor, or score tensor.
    N : int
        Number of nodes in the output mask.

    Returns
    -------
    torch.Tensor
        Boolean mask of length ``N``.
    """
    mask = mask.view(-1)

    if mask.dtype == torch.bool and mask.numel() == N:
        return mask

    if mask.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        out = torch.zeros(N, dtype=torch.bool)
        if mask.numel() > 0:
            out[mask.long()] = True
        return out

    if mask.numel() == N:
        return mask != 0

    return torch.zeros(N, dtype=torch.bool)


def _tensor_schema_entry(t: torch.Tensor) -> dict[str, Any]:
    """Create an on-disk schema entry for a tensor value.

    Parameters
    ----------
    t : torch.Tensor
        Tensor to describe in an :class:`OnDiskDataset` schema.

    Returns
    -------
    dict[str, Any] or type
        Schema entry describing the tensor dtype and trailing shape, or a scalar
        Python type for zero-dimensional tensors.
    """
    if t.dim() == 0:
        if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return int
        if t.dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ):
            return float
        if t.dtype == torch.bool:
            return bool
        return dict(dtype=t.dtype, size=(-1,))
    size = (-1,) + tuple(int(d) for d in t.size()[1:])
    return dict(dtype=t.dtype, size=size)


class ClusterOnDisk(OnDiskDataset):
    """On-disk storage and metadata for Cluster-GCN training.

    Builds a global partition using METIS or Random partitioning, infers a
    generic schema over all cluster subgraphs, stores them on disk, and
    writes permuted structural and feature arrays as NumPy memmaps.

    Parameters
    ----------
    root : str
        Root directory for the on-disk dataset.
    graph_getter : callable
        Callable returning the full :class:`torch_geometric.data.Data` graph.
    num_parts : int, optional
        Number of clusters for partitioning. Default is 10.
    partition_method : {"metis", "random"}, optional
        Strategy used to cluster nodes. Default is "metis".
    partition_seed : int, optional
        Seed for random partitioning. If ``None``, a fresh random seed is used.
    recursive : bool, optional
        Whether to apply recursive partitioning (METIS only). Default is False.
    keep_inter_cluster_edges : bool, optional
        If True, inter-cluster edges are kept. Default is False.
    sparse_format : {"csr"}, optional
        Sparse adjacency representation. Default is "csr".
    backend : {"sqlite", "rocksdb"}, optional
        On-disk backend. Default is "sqlite".
    transform : callable, optional
        Transform applied on loaded cluster subgraphs.
    pre_filter : callable, optional
        Filter applied before writing samples to disk.
    """

    def __init__(
        self,
        root: str,
        *,
        graph_getter: Callable[[], Data],
        num_parts: int = 10,
        partition_method: str = "metis",
        partition_seed: int | None = None,
        recursive: bool = False,
        keep_inter_cluster_edges: bool = False,
        sparse_format: str = "csr",
        backend: str = "sqlite",
        transform=None,
        pre_filter=None,
    ) -> None:
        self._graph_getter = graph_getter
        self._cfg = dict(
            num_parts=int(num_parts),
            partition_method=str(partition_method).lower(),
            partition_seed=partition_seed,
            recursive=bool(recursive),
            keep_inter=bool(keep_inter_cluster_edges),
            sparse_format=str(sparse_format),
        )

        full = self._graph_getter()

        # Handle partitioning strategy selection
        if self._cfg["partition_method"] == "metis":
            cluster_data = ClusterData(
                full,
                num_parts=self._cfg["num_parts"],
                recursive=self._cfg["recursive"],
                keep_inter_cluster_edges=self._cfg["keep_inter"],
                sparse_format=self._cfg["sparse_format"],
                save_dir=None,
                log=False,
            )
        elif self._cfg["partition_method"] == "random":
            cluster_data = self._build_random_partition(full)
        else:
            raise ValueError(
                f"Unknown partition_method: {self._cfg['partition_method']}. "
                f"Choose 'metis' or 'random'."
            )

        # Discover schema across ALL parts:
        discovered: dict[str, Any] = {
            "edge_index": dict(dtype=torch.long, size=(2, -1))
        }

        for i in range(len(cluster_data)):
            part = cluster_data[i]
            if getattr(part, "edge_index", None) is None:
                raise ValueError(
                    "Cluster part without edge_index; cannot store."
                )
            for key, val in self._iter_data_items(part):
                if key == "edge_index":
                    continue
                self._schema_union_update(discovered, key, val)

        self._bootstrap_full = full
        self._bootstrap_cluster_data = cluster_data

        super().__init__(
            root,
            transform=transform,
            pre_filter=pre_filter,
            backend=backend,
            schema=discovered,
        )
        self._meta: dict[str, Any] | None = None

    def _build_random_partition(self, data: Data) -> ClusterData:
        """Build a ClusterData-compatible object from random node assignments.

        Generates a random node clustering assignment, then reuses PyG's
        partition/permutation helpers so downstream ClusterData indexing sees
        the same internal structure as the METIS path.

        Parameters
        ----------
        data : Data
            Full graph to partition.

        Returns
        -------
        ClusterData
            ClusterData-compatible wrapper with a random node partition.
        """
        if data.edge_index is None:
            raise ValueError("Cannot partition graph without edge_index.")
        if data.num_nodes is None:
            raise ValueError("Cannot infer num_nodes for random partitioning.")

        num_nodes = int(data.num_nodes)
        num_parts = self._cfg["num_parts"]

        generator = torch.Generator(device=data.edge_index.device)
        if self._cfg["partition_seed"] is not None:
            generator.manual_seed(int(self._cfg["partition_seed"]))
        else:
            generator.seed()

        node_perm = torch.randperm(
            num_nodes,
            generator=generator,
            device=data.edge_index.device,
        )
        cluster_id = torch.empty(
            num_nodes,
            dtype=torch.long,
            device=data.edge_index.device,
        )
        cluster_id[node_perm] = (
            torch.arange(num_nodes, device=data.edge_index.device) * num_parts
        ) // num_nodes

        cluster_data = ClusterData.__new__(ClusterData)
        cluster_data.num_parts = num_parts
        cluster_data.recursive = self._cfg["recursive"]
        cluster_data.keep_inter_cluster_edges = self._cfg["keep_inter"]
        cluster_data.sparse_format = self._cfg["sparse_format"]
        cluster_data.partition = cluster_data._partition(
            data.edge_index,
            cluster_id,
        )
        cluster_data.data = cluster_data._permute_data(
            data,
            cluster_data.partition,
        )

        return cluster_data

    @property
    def raw_file_names(self) -> list[str]:
        """Return raw file names required by the dataset.

        Returns
        -------
        list[str]
            Empty list because the full graph is supplied by ``graph_getter``.
        """
        return []

    def download(self) -> None:
        """Skip downloading because data is supplied by ``graph_getter``."""

    @staticmethod
    def _schema_union_update(
        schema: dict[str, Any], key: str, val: Any
    ) -> None:
        """Update an on-disk schema with one serialized value.

        Parameters
        ----------
        schema : dict[str, Any]
            Mutable schema dictionary to update.
        key : str
            Attribute name in the serialized data object.
        val : Any
            Attribute value used to infer the schema entry.
        """
        if val is None:
            return
        if isinstance(val, torch.Tensor):
            entry = _tensor_schema_entry(val)
            schema.setdefault(key, entry)
        elif isinstance(val, (int, bool, float)):
            schema.setdefault(key, type(val))

    @staticmethod
    def _iter_data_items(d: Data):
        """Iterate over serializable items in a data object.

        Parameters
        ----------
        d : Data
            Data object to inspect.

        Yields
        ------
        tuple[str, Any]
            Attribute name and value pairs.
        """
        for k in d.keys():  # noqa: SIM118
            yield k, getattr(d, k)
        if getattr(d, "num_nodes", None) is not None:
            yield "num_nodes", int(d.num_nodes)

    def process(self) -> None:
        """Partition the graph and persist cluster samples plus memmaps."""
        full: Data = self._bootstrap_full
        cluster_data: ClusterData = self._bootstrap_cluster_data

        buf: list[Data] = []
        for i in range(len(cluster_data)):
            buf.append(cluster_data[i])
            if (i + 1) % 1000 == 0 or (i + 1) == len(cluster_data):
                self.extend(buf)
                buf = []

        meta = {
            "num_parts": cluster_data.num_parts,
            "partition_method": self._cfg["partition_method"],
            "partition_seed": self._cfg["partition_seed"],
            "recursive": cluster_data.recursive
            if hasattr(cluster_data, "recursive")
            else False,
            "keep_inter_cluster_edges": cluster_data.keep_inter_cluster_edges,
            "sparse_format": cluster_data.sparse_format,
            "partition": cluster_data.partition,
        }
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(meta, self._meta_path)

        self._write_perm_memmaps(full, cluster_data.partition)

        self._bootstrap_full = None
        self._bootstrap_cluster_data = None

    def serialize(self, data: Data) -> dict[str, Any]:
        """Serialize one cluster data object for on-disk storage.

        Parameters
        ----------
        data : Data
            Cluster subgraph to serialize.

        Returns
        -------
        dict[str, Any]
            Dictionary matching the inferred on-disk schema.
        """
        row: dict[str, Any] = {}
        if getattr(data, "edge_index", None) is None:
            raise ValueError(
                "Data object without edge_index cannot be serialized."
            )
        row["edge_index"] = data.edge_index

        for key in self.schema:
            if key == "edge_index":
                continue
            if hasattr(data, key):
                val = getattr(data, key)
                if isinstance(val, (torch.Tensor, int, bool, float)):
                    row[key] = val
        return row

    def deserialize(self, row: dict[str, Any]) -> Data:
        """Deserialize one on-disk row into a data object.

        Parameters
        ----------
        row : dict[str, Any]
            Serialized row loaded from the on-disk backend.

        Returns
        -------
        Data
            Reconstructed cluster subgraph.
        """
        return Data.from_dict(row)

    @property
    def _meta_path(self) -> str:
        """Return the path to the cluster metadata file.

        Returns
        -------
        str
            Cluster metadata path.
        """
        return osp.join(self.processed_dir, "cluster_meta.pt")

    @property
    def meta(self) -> dict[str, Any]:
        """Return cached cluster metadata.

        Returns
        -------
        dict[str, Any]
            Metadata loaded from ``cluster_meta.pt``.
        """
        if self._meta is None:
            self._meta = torch.load(
                self._meta_path, map_location="cpu", weights_only=False
            )
        return self._meta

    @property
    def partition(self):
        """Return the PyG partition object.

        Returns
        -------
        Any
            Partition object containing node, edge, and part pointers.
        """
        return self.meta["partition"]

    @property
    def num_parts(self) -> int:
        """Return the number of partition parts.

        Returns
        -------
        int
            Number of graph clusters.
        """
        return int(self.meta["num_parts"])

    @property
    def partition_method(self) -> str:
        """Return the partitioning method.

        Returns
        -------
        str
            Partitioning method name.
        """
        return str(self.meta.get("partition_method", "metis"))

    @property
    def recursive(self) -> bool:
        """Return whether recursive partitioning was used.

        Returns
        -------
        bool
            Recursive partitioning flag.
        """
        return bool(self.meta["recursive"])

    @property
    def keep_inter_cluster_edges(self) -> bool:
        """Return whether inter-cluster edges were preserved.

        Returns
        -------
        bool
            Inter-cluster edge preservation flag.
        """
        return bool(self.meta["keep_inter_cluster_edges"])

    @property
    def sparse_format(self) -> str:
        """Return the sparse format used for the partition.

        Returns
        -------
        str
            Sparse adjacency format.
        """
        return str(self.meta["sparse_format"])

    def _memmap_dir(self) -> str:
        """Return the memmap output directory.

        Returns
        -------
        str
            Path containing permuted arrays and CSR memmaps.
        """
        return osp.join(self.processed_dir, "perm_memmap")

    def _write_perm_memmaps(self, full: Data, P: Any) -> None:
        """Write permuted graph arrays for streaming dataloaders.

        Parameters
        ----------
        full : Data
            Full graph before partition permutation.
        P : Any
            PyG partition object containing node and edge permutations.
        """
        out_dir = self._memmap_dir()
        os.makedirs(out_dir, exist_ok=True)

        np.save(osp.join(out_dir, "partptr.npy"), P.partptr.cpu().numpy())
        np.save(osp.join(out_dir, "indptr.npy"), P.indptr.cpu().numpy())
        np.save(osp.join(out_dir, "indices.npy"), P.index.cpu().numpy())

        node_perm = P.node_perm.cpu()
        N = int(full.num_nodes)

        perm_to_global = node_perm.clone().to(torch.long)
        np.save(
            osp.join(out_dir, "perm_to_global.npy"), perm_to_global.numpy()
        )

        global_to_perm = torch.empty_like(perm_to_global)
        global_to_perm[perm_to_global] = torch.arange(
            perm_to_global.numel(), dtype=torch.long
        )
        np.save(
            osp.join(out_dir, "global_to_perm.npy"), global_to_perm.numpy()
        )

        if getattr(full, "x", None) is not None and full.x.numel() > 0:
            x = full.x
            if x.dim() == 1:
                x = x.view(-1, 1)
            F = int(x.size(1))
            X_path = osp.join(out_dir, "X_perm.npy")
            X_mm = open_memmap(
                X_path, dtype="float32", mode="w+", shape=(N, F)
            )
            X_mm[:] = x[node_perm].to(torch.float32).cpu().numpy()
            del X_mm

        if getattr(full, "y", None) is not None:
            y_src = full.y.view(-1)[node_perm].to(torch.int64).cpu().numpy()
            y_path = osp.join(out_dir, "y_perm.npy")
            y_mm = open_memmap(
                y_path, dtype="int64", mode="w+", shape=(y_src.shape[0],)
            )
            y_mm[:] = y_src
            del y_mm

        if getattr(full, "edge_attr", None) is not None:
            ea = full.edge_attr
            if ea.dim() == 1:
                ea = ea.view(-1, 1)
            ea_src = ea[P.edge_perm].to(torch.float32).cpu().numpy()
            E, F_e = ea_src.shape
            ea_path = osp.join(out_dir, "edge_attr_perm.npy")
            ea_mm = open_memmap(
                ea_path, dtype="float32", mode="w+", shape=(E, F_e)
            )
            ea_mm[:] = ea_src
            del ea_mm
