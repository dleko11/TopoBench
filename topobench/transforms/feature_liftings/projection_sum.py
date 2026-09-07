"""ProjectionSum class."""

import torch
import torch_geometric

from topobench.utils.phase_tracking import track_phase


def _projection_phase(element: str) -> str:
    """Return the tracked phase for one projected structure rank.

    Parameters
    ----------
    element : str
        Structure rank or ``hyperedges`` identifier.

    Returns
    -------
    str
        Phase name used by the resource tracker.
    """
    if element == "hyperedges":
        return "feature_projection_hyperedges"
    if element in {"1", "2", "3"}:
        return f"feature_projection_rank_{element}"
    return "feature_projection_other"


class ProjectionSum(torch_geometric.transforms.BaseTransform):
    r"""Lift r-cell features to r+1-cells by projection.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Project r-cell features of a graph to r+1-cell structures.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The data with the lifted features.
        """
        keys = sorted(
            [
                key.split("_")[1]
                for key in data
                if ("incidence" in key and "-" not in key)
            ]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                with track_phase(
                    _projection_phase(elem),
                    extra={"tracking/feature_projection_rank": elem},
                ):
                    data["x_" + elem] = torch.matmul(
                        abs(data["incidence_" + elem].t().float()),
                        data[f"x_{idx_to_project}"].float(),
                    )
        return data

    def forward(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Apply the lifting to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The lifted data.
        """
        data = self.lift_features(data)
        return data
