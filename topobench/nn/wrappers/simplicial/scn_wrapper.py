"""Wrapper for the SCNW model."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class SCNWrapper(AbstractWrapper):
    r"""Wrapper for the SCNW model.

    This wrapper defines the forward pass of the model. The SCNW model returns
    the embeddings of the cells of rank 0, 1, and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the SCNW wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        laplacian_0 = self.normalize_matrix(batch.hodge_laplacian_0)
        laplacian_1 = self.normalize_matrix(batch.hodge_laplacian_1)
        laplacian_2 = self.normalize_matrix(batch.hodge_laplacian_2)
        x_0, x_1, x_2 = self.backbone(
            batch.x_0,
            batch.x_1,
            batch.x_2,
            laplacian_0,
            laplacian_1,
            laplacian_2,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_2"] = x_2
        model_out["x_1"] = x_1
        model_out["x_0"] = x_0

        return model_out

    def normalize_matrix(self, matrix):
        r"""Normalize the input matrix.

        The normalization is performed using the diagonal matrix of the inverse square root of the sum of the absolute values of the rows.

        Parameters
        ----------
        matrix : torch.sparse.FloatTensor
            Input matrix to be normalized.

        Returns
        -------
        torch.sparse.FloatTensor
            Normalized matrix.
        """
        if not matrix.is_sparse or matrix.layout != torch.sparse_coo:
            matrix = matrix.to_sparse_coo()

        matrix = matrix.coalesce()
        indices = matrix.indices()
        values = matrix.values()
        n_rows, n_cols = matrix.shape

        if n_rows != n_cols:
            raise ValueError(
                "SCN Hodge Laplacian normalization expects square matrices, "
                f"got shape {tuple(matrix.shape)}."
            )

        row_sums = torch.zeros(
            n_rows, dtype=values.dtype, device=values.device
        )
        if values.numel() > 0:
            row_sums.scatter_add_(0, indices[0], values.abs())

        inv_sqrt = torch.zeros_like(row_sums)
        nonzero = row_sums != 0
        inv_sqrt[nonzero] = torch.rsqrt(row_sums[nonzero])

        normalized_values = (
            values * inv_sqrt[indices[0]] * inv_sqrt[indices[1]]
        )
        return torch.sparse_coo_tensor(
            indices,
            normalized_values,
            matrix.shape,
            device=matrix.device,
            dtype=values.dtype,
        ).coalesce()
