"""Comprehensive test suite for on-disk dataset loaders."""

from typing import Any
from pathlib import Path
import os
import hydra
import pytest
import torch_geometric
from torch_geometric.data import OnDiskDataset

# Exclude datasets to keep CI fast and stable
EXCLUDE_DATASETS = {
    "karate_club.yaml",
    "REDDIT-BINARY.yaml",
    "IMDB-MULTI.yaml",
    "IMDB-BINARY.yaml",
    "ogbg-molpcba.yaml",
    "manual_dataset.yaml",
}

# Identify long running datasets
LONG_RUNNING_DATASETS = {
    "mantra_name.yaml",
    "mantra_orientation.yaml",
    "mantra_genus.yaml",
    "mantra_betti_numbers.yaml",
}


def gather_config_files() -> list[tuple[str, str]]:
    """Gather all dataset configuration files from the configs/dataset folder.

    Returns
    -------
    list of tuple of str
        List of (data_domain, config_file) pairs.
    """
    config_files = []
    base_dir = Path(__file__).resolve().parents[3]
    config_base_dir = base_dir / "configs/dataset"

    if not config_base_dir.exists():
        return [("graph", "fake.yaml")]

    for dir_path in config_base_dir.iterdir():
        curr_dir = dir_path.name
        if dir_path.is_dir():
            for f in dir_path.glob("*.yaml"):
                if f.name in EXCLUDE_DATASETS:
                    continue
                config_files.append((curr_dir, f.name))

    return config_files


# Gather config files for parameterization
CONFIG_FILES = gather_config_files()


@pytest.fixture(autouse=True)
def setup_hydra():
    """Clear global Hydra instance before each test.

    Yields
    ------
    None
        Does not yield any value.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    yield
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def load_dataset(data_domain: str, config_file: str) -> tuple[OnDiskDataset | None, dict, Any]:
    """Load dataset and configuration for the given YAML file.

    Parameters
    ----------
    data_domain : str
        Dataset domain name.
    config_file : str
        Dataset config filename.

    Returns
    -------
    OnDiskDataset or None
        Loaded on-disk dataset or None if memory_type is not on_disk.
    dict
        Dataset directory metadata as returned by the loader.
    Any
        Full Hydra-composed configuration.
    """
    relative_config_dir = "../../../configs"
    with hydra.initialize(
        version_base="1.3",
        config_path=relative_config_dir,
        job_name="run",
    ):
        parameters = hydra.compose(
            config_name="run.yaml",
            overrides=[
                f"dataset={data_domain}/{config_file}",
                "model=graph/gat",
            ],
            return_hydra_config=True,
        )

        memory_type = parameters.dataset.loader.parameters.get(
            "memory_type", "in_memory"
        )

        if memory_type != "on_disk":
            return None, {}, parameters

        dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)

        if config_file in LONG_RUNNING_DATASETS:
            dataset, data_dir = dataset_loader.load(slice=100)
        else:
            dataset, data_dir = dataset_loader.load()

    return dataset, data_dir, parameters


@pytest.mark.parametrize("data_domain,config_file", CONFIG_FILES)
def test_on_disk_dataset_loading(data_domain: str, config_file: str):
    """Test loading and verify basic properties for on-disk datasets.

    For configs with memory_type == 'on_disk', this verifies that the loader
    returns a valid OnDiskDataset, and that features/labels are non-empty
    and consistent.

    Parameters
    ----------
    data_domain : str
        The data domain subdirectory name.
    config_file : str
        The dataset configuration filename.
    """
    dataset, _, _ = load_dataset(data_domain, config_file)

    # Skip in-memory datasets
    if dataset is None:
        pytest.skip(f"Skipping in-memory dataset configuration: {config_file}")

    # Check that the returned dataset is backed by OnDiskDataset
    assert isinstance(dataset, OnDiskDataset) or isinstance(
        dataset, torch_geometric.data.OnDiskDataset
    )

    # Dataset must contain at least one graph
    assert len(dataset) > 0

    # Single-graph style (dataset.data) or multi-graph style (dataset[0])
    if hasattr(dataset, "data"):
        data = dataset.data
    else:
        data = dataset[0]

    # Basic feature and label checks
    assert hasattr(data, "x"), "Missing node features"
    assert hasattr(data, "y"), "Missing labels"
    assert data.x is not None and data.x.numel() > 0, "Empty node features"
    assert data.y is not None and data.y.numel() > 0, "Empty labels"

    # Node feature dimension consistency when available
    if hasattr(dataset, "num_node_features"):
        assert data.x.size(1) == dataset.num_node_features

    # Basic repr should not crash
    assert repr(dataset) is not None
