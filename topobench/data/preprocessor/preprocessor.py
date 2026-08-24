"""Preprocessor for datasets."""

import copy
import hashlib
import json
import logging
import os
import os.path as osp
import shutil
import time
from typing import Any

import filelock
import numpy as np
import torch
import torch_geometric
from filelock import FileLock
from torch_geometric.io import fs
from tqdm import tqdm

from topobench.data.utils import (
    ClusterOnDisk,
    ensure_serializable,
    load_inductive_splits,
    load_transductive_splits,
    make_hash,
    to_bool_mask,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


def _split_mask_fingerprint(masks: dict[str, torch.Tensor]) -> str:
    """Return a stable fingerprint for effective train/val/test masks.

    Parameters
    ----------
    masks : dict[str, torch.Tensor]
        Boolean-compatible train, validation, and test masks.

    Returns
    -------
    str
        Stable fingerprint of the effective split masks.
    """
    digest = hashlib.sha1()
    for name in ("train", "val", "test"):
        mask = masks[name].detach().to(device="cpu", dtype=torch.bool)
        values = mask.contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


class PreProcessor(torch_geometric.data.InMemoryDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : list
        List of data objects.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset
        self.data_dir = data_dir
        self.preprocessing_time = 0
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )

            # 1. Ensure the target directory exists so we can place a lock file in it
            os.makedirs(self.processed_data_dir, exist_ok=True)
            lock_path = os.path.join(
                self.processed_data_dir, "preprocessing.lock"
            )

            start_time = time.time()

            with FileLock(lock_path):
                # When Process 1 finishes, Process 2 checks, sees data.pt, and skips.
                super().__init__(
                    self.processed_data_dir, None, pre_transform, **kwargs
                )
                self.save_transform_parameters()

            end_time = time.time()
            self.preprocessing_time = end_time - start_time

            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.load(self.processed_paths[0])
            self.data_list = [data for data in self]
        else:
            self.transforms_applied = False
            super().__init__(data_dir, None, None, **kwargs)
            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.data, self.slices = dataset._data, dataset.slices
            self.data_list = [data for data in dataset]

        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx
        if hasattr(dataset, "split_idx_list"):
            self.split_idx_list = dataset.split_idx_list

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return self.root

    @property
    def processed_file_names(self) -> str:
        """Return the name of the processed file.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        from torch_geometric.transforms import ToDevice

        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings

        if "transform_name" in transforms_config:
            config_items = [
                (transforms_config.transform_name, transforms_config)
            ]
        else:
            config_items = transforms_config.items()

        pre_transforms_list = []
        pre_transforms_dict = {}

        # Track where the graph currently lives in the pipeline
        current_device = "cpu"

        for key, value in config_items:
            kwargs = dict(value)

            requested_device = kwargs.pop("preprocessor_device", "cpu")

            target_device = (
                "cuda"
                if requested_device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )

            transform_instance = DataTransform(**kwargs)
            pre_transforms_dict[key] = transform_instance

            if target_device != current_device:
                pre_transforms_list.append(ToDevice(target_device))
                current_device = target_device

            pre_transforms_list.append(transform_instance)

        # If the pipeline ends while the graph is still on the GPU,
        # we MUST pull it back to the CPU before PyTorch Geometric saves it to disk.
        if current_device == "cuda":
            pre_transforms_list.append(ToDevice("cpu"))

        pre_transforms = torch_geometric.transforms.Compose(
            pre_transforms_list
        )

        self.set_processed_data_dir(
            pre_transforms_dict, data_dir, transforms_config
        )
        return pre_transforms

    def set_processed_data_dir(
        self, pre_transforms_dict, data_dir, transforms_config
    ) -> None:
        """Set the processed data directory.

        Parameters
        ----------
        pre_transforms_dict : dict
            Dictionary containing the pre-transforms.
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        """Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
            )

    def process(self) -> None:
        """Method that processes the data."""
        if isinstance(
            self.dataset,
            (torch_geometric.data.Dataset, torch.utils.data.Dataset),
        ):
            data_list = [data for data in self.dataset]
        elif isinstance(self.dataset, torch_geometric.data.Data):
            data_list = [self.dataset]

        if self.pre_transform is not None:
            print(f"\nApplying transforms to {len(data_list)} graphs...")
            self.data_list = [
                self.pre_transform(d)
                for d in tqdm(
                    data_list, desc="Processing graphs", unit="graph"
                )
            ]
        else:
            self.data_list = data_list

        self._data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])

    def load(self, path: str) -> None:
        r"""Load the dataset from the file path `path`.

        Parameters
        ----------
        path : str
            The path to the processed data.
        """
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) >= 2 and len(out) <= 4
        if len(out) == 2:  # Backward compatibility (1).
            data, self.slices = out
        elif len(out) == 3:  # Backward compatibility (2).
            data, self.slices, data_cls = out
        else:  # TU Datasets store additional element (__class__) in the processed file
            data, self.slices, sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load the dataset splits.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )

    def pack_global_partition(
        self,
        split_params: dict,
        cluster_params: dict,
        stream_params: dict,
        dtype_policy: str = "preserve",
        pack_db: bool = True,
        pack_memmaps: bool = True,
    ) -> dict[str, Any]:
        """Build and persist a global Cluster-GCN partition.

        The returned handle dictionary contains paths and metadata required by
        block-streaming dataloaders (e.g. `TBBlockStreamDataModule`) to build
        Cluster-GCN-style mini-batches directly from disk without reloading the
        full graph into memory.

        Parameters
        ----------
        split_params : dict
            Parameters for the split pipeline; must define a transductive
            single-graph setting and produce train/val/test masks.
        cluster_params : dict
            Parameters controlling graph partitioning:
            `num_parts`, `recursive`, `keep_inter_cluster_edges`,
            `sparse_format`, etc.
        stream_params : dict
            Parameters for downstream streaming. These do not affect the
            global partition identity.
        dtype_policy : {"preserve", "float32"}, optional
            Policy for persisting feature/edge_attr dtypes. Recorded in meta
            for downstream consumers.
        pack_db : bool, optional
            If True, keep the `OnDiskDataset` DB of per-cluster subgraphs.
        pack_memmaps : bool, optional
            If True, write CSR and permuted feature/label/mask memmaps.

        Returns
        -------
        dict
            A handle with root/processed/memmap paths, partition metadata, and
            file locations for all relevant arrays.
        """
        root = self.data_dir
        processed_base = osp.join(root, "processed")
        os.makedirs(processed_base, exist_ok=True)

        # Split creation writes a shared set of split files. Serialize it so
        # concurrent seeds cannot observe a partially generated split set.
        split_generation_lock = osp.join(processed_base, "splits.lock")
        with filelock.FileLock(split_generation_lock, timeout=-1):
            dataset_train, _, _ = self.load_dataset_splits(split_params)

        # Always use the split returned by the split pipeline. This is
        # important for datasets such as Reddit and Planetoid, whose raw PyG
        # objects already contain fixed masks that must not override a
        # requested random split.
        full = dataset_train.data_lst[0]

        if getattr(full, "num_nodes", None) is not None:
            num_nodes = int(full.num_nodes)
        elif getattr(full, "x", None) is not None:
            num_nodes = int(full.x.size(0))
            full.num_nodes = num_nodes
        elif getattr(full, "y", None) is not None:
            num_nodes = int(full.y.size(0))
            full.num_nodes = num_nodes
        else:
            raise ValueError("Cannot infer num_nodes from full graph.")

        masks = {
            "train": to_bool_mask(full.train_mask, num_nodes),
            "val": to_bool_mask(full.val_mask, num_nodes),
            "test": to_bool_mask(full.test_mask, num_nodes),
        }
        full.train_mask = masks["train"]
        full.val_mask = masks["val"]
        full.test_mask = masks["test"]
        split_hash = _split_mask_fingerprint(masks)

        if getattr(full, "edge_index", None) is None:
            raise ValueError("Full graph has no edge_index.")

        # Split masks do not affect the graph partition. Keep one structural
        # partition and attach small split-specific mask sidecars. Feature
        # standardization is split-dependent, so standardized datasets retain
        # separate structural partitions.
        cluster_config = {
            "format_version": 2,
            "cluster_params": ensure_serializable(cluster_params),
            "dtype_policy": dtype_policy,
        }
        if split_params.get("standardize", False):
            cluster_config["standardized_split_hash"] = split_hash
        config_hash = make_hash(cluster_config)

        part_dir = osp.join(processed_base, f"part_{config_hash}")
        structural_handle_path = osp.join(part_dir, "structural_handle.pt")
        lock_path = osp.join(processed_base, f"part_{config_hash}.lock")
        split_base = osp.join(part_dir, "splits")
        split_dir = osp.join(split_base, f"split_{split_hash}")
        handle_path = osp.join(split_dir, "handle.pt")
        split_lock_path = osp.join(split_base, f"split_{split_hash}.lock")

        if osp.exists(handle_path):
            logging.info(
                "[pack_global_partition] Reusing cached partition and split: "
                f"{part_dir} (split={split_hash})"
            )
            return torch.load(
                handle_path, map_location="cpu", weights_only=False
            )

        with filelock.FileLock(lock_path, timeout=-1):
            if not osp.exists(structural_handle_path):
                if osp.isdir(part_dir):
                    shutil.rmtree(part_dir)
                os.makedirs(part_dir, exist_ok=True)
                logging.info(
                    "[pack_global_partition] Building structural partition "
                    f"(hash={config_hash}): {part_dir}"
                )

                structural_full = copy.copy(full)
                for mask_name in ("train_mask", "val_mask", "test_mask"):
                    if mask_name in structural_full:
                        del structural_full[mask_name]

                num_parts = int(cluster_params.get("num_parts", 10))
                recursive = bool(cluster_params.get("recursive", False))
                keep_inter = bool(
                    cluster_params.get("keep_inter_cluster_edges", False)
                )
                sparse_format = str(cluster_params.get("sparse_format", "csr"))
                ds = ClusterOnDisk(
                    root=part_dir,
                    graph_getter=lambda: structural_full,
                    num_parts=num_parts,
                    recursive=recursive,
                    keep_inter_cluster_edges=keep_inter,
                    sparse_format=sparse_format,
                    backend="sqlite",
                    transform=None,
                    pre_filter=None,
                )
                _ = len(ds)
                torch.save(ds.schema, osp.join(ds.processed_dir, "schema.pt"))

                mm_dir = osp.join(ds.processed_dir, "perm_memmap")
                structural_handle = {
                    "root": ds.root,
                    "processed_dir": ds.processed_dir,
                    "memmap_dir": mm_dir,
                    "num_parts": int(ds.num_parts),
                    "sparse_format": str(ds.sparse_format),
                    "has_x": getattr(full, "x", None) is not None,
                    "has_y": getattr(full, "y", None) is not None,
                    "has_edge_attr": getattr(full, "edge_attr", None)
                    is not None,
                    "config_hash": config_hash,
                    "paths": {
                        "partptr": osp.join(mm_dir, "partptr.npy"),
                        "indptr": osp.join(mm_dir, "indptr.npy"),
                        "indices": osp.join(mm_dir, "indices.npy"),
                        "perm_to_global": osp.join(
                            mm_dir, "perm_to_global.npy"
                        ),
                        "global_to_perm": osp.join(
                            mm_dir, "global_to_perm.npy"
                        ),
                        "X_perm": osp.join(mm_dir, "X_perm.npy"),
                        "y_perm": osp.join(mm_dir, "y_perm.npy"),
                        "edge_attr_perm": osp.join(
                            mm_dir, "edge_attr_perm.npy"
                        ),
                    },
                }
                tmp_structural_handle = structural_handle_path + ".tmp"
                torch.save(structural_handle, tmp_structural_handle)
                os.replace(tmp_structural_handle, structural_handle_path)
            else:
                logging.info(
                    "[pack_global_partition] Reusing structural partition: "
                    f"{part_dir}"
                )

        structural_handle = torch.load(
            structural_handle_path, map_location="cpu", weights_only=False
        )
        os.makedirs(split_base, exist_ok=True)
        with filelock.FileLock(split_lock_path, timeout=-1):
            if osp.exists(handle_path):
                return torch.load(
                    handle_path, map_location="cpu", weights_only=False
                )

            if osp.isdir(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir, exist_ok=True)
            logging.info(
                f"[pack_global_partition] Writing split sidecars: {split_dir}"
            )

            mm_dir = structural_handle["memmap_dir"]
            node_perm = np.load(
                structural_handle["paths"]["perm_to_global"]
                if "perm_to_global" in structural_handle["paths"]
                else osp.join(mm_dir, "perm_to_global.npy")
            )
            partptr = np.load(structural_handle["paths"]["partptr"])

            def _to_numpy_bool(mask: torch.Tensor) -> np.ndarray:
                return mask.view(-1)[node_perm].to(torch.bool).cpu().numpy()

            mask_paths: dict[str, str] = {}
            parts_paths: dict[str, str] = {}
            for split_name, mask in masks.items():
                mask_perm = _to_numpy_bool(mask)
                mask_path = osp.join(split_dir, f"{split_name}_mask_perm.npy")
                np.save(mask_path, mask_perm)
                mask_paths[split_name] = mask_path

                positions = np.flatnonzero(mask_perm)
                part_ids = (
                    np.searchsorted(partptr, positions, side="right") - 1
                )
                parts_path = osp.join(
                    split_dir, f"parts_with_{split_name}.npy"
                )
                np.save(parts_path, np.unique(part_ids.astype(np.int64)))
                parts_paths[split_name] = parts_path

            handle = dict(structural_handle)
            handle["paths"] = dict(structural_handle["paths"])
            for split_name in ("train", "val", "test"):
                handle["paths"][f"{split_name}_mask_perm"] = mask_paths[
                    split_name
                ]
                handle["paths"][f"parts_with_{split_name}"] = parts_paths[
                    split_name
                ]
            handle["split_hash"] = split_hash

            tmp_handle = handle_path + ".tmp"
            torch.save(handle, tmp_handle)
            os.replace(tmp_handle, handle_path)
            return handle
