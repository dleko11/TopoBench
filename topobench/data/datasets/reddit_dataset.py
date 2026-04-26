"""Reddit dataset wrapper."""

import os.path as osp

from torch_geometric.datasets import Reddit as PyGReddit


class RedditDataset(PyGReddit):
    """Thin wrapper around :class:`torch_geometric.datasets.Reddit`.

    Parameters
    ----------
    root : str
        Root directory for all datasets.
    name : str
        Dataset name used as a subdirectory under ``root``.
    **kwargs
        Additional keyword arguments passed to
        :class:`torch_geometric.datasets.Reddit`.
    """

    def __init__(self, root: str, name: str, **kwargs) -> None:
        self.name = name
        super().__init__(root=root, **kwargs)

    @property
    def raw_dir(self) -> str:
        """Path to the raw data directory.

        Returns
        -------
        str
            Path to the raw data directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Path to the processed data directory.

        Returns
        -------
        str
            Path to the processed data directory.
        """
        return osp.join(self.root, self.name, "processed")
