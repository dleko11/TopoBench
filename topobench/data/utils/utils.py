"""Data utilities."""

import hashlib

import networkx as nx
import numpy as np
import omegaconf
import time
import torch
import torch_geometric
import torch_geometric.utils
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import SimplicialComplex


def fast_sparse_mm(A, B, label=""):
    """Fast sparse matrix multiplication using native PyTorch.

    Parameters
    ----------
    A : torch.Tensor (sparse)
        Left matrix.
    B : torch.Tensor (sparse)
        Right matrix.
    label : str, optional
        Label for debugging.

    Returns
    -------
    torch.Tensor (sparse)
        Result of A @ B.
    """
    return torch.sparse.mm(A.coalesce(), B.coalesce())


def assemble_connectivity(B, layout="up"):
    """Assemble connectivity matrix from incidence matrix B.

    This is a memory-efficient replacement for torch.sparse.mm(B, B.T) or
    torch.sparse.mm(B.T, B) specifically for simplicial incidence matrices.
    It bypasses general sparse @ sparse kernels which can have massive
    memory overhead/spikes.

    Parameters
    ----------
    B : torch.Tensor (sparse)
        Incidence matrix of size (nf, ns).
    layout : str, optional
        "up" for B @ B.T (nf x nf),
        "down" for B.T @ B (ns x ns).

    Returns
    -------
    torch.Tensor (sparse)
        The assembled connectivity matrix.
    """
    B = B.coalesce()
    nf, ns = B.size()

    if layout == "up":
        # Compute L = B @ B.T (Faces x Faces)
        # To avoid the 1.4 TB spike, we process simplices (columns) in chunks.
        # B @ B.T = sum over chunks of (B_chunk @ B_chunk.T)
        chunk_size = 50000  # Adjust based on available RAM, 50k is very safe
        res = None
        
        # If the matrix is small, just do it in one go
        if ns <= chunk_size:
            return torch.sparse.mm(B, B.t()).coalesce()

        indices = B.indices()
        values = B.values()
        
        for i in range(0, ns, chunk_size):
            start = i
            length = min(chunk_size, ns - start)
            
            # Manual slicing of coalesced matrix for speed and memory efficiency
            mask = (indices[1] >= start) & (indices[1] < start + length)
            
            if not mask.any():
                continue
                
            chunk_indices = indices[:, mask].clone()
            chunk_indices[1] -= start # Re-index to local chunk columns
            chunk_values = values[mask].clone()
            
            B_chunk = torch.sparse_coo_tensor(
                chunk_indices, 
                chunk_values,
                size=(nf, length),
                device=B.device
            ).coalesce()
            
            # Compute partial product
            prod = torch.sparse.mm(B_chunk, B_chunk.t()).coalesce()
            
            if res is None:
                res = prod
            else:
                # Accumulate sparse matrices
                res = (res + prod).coalesce()
        
        return res if res is not None else torch.sparse_coo_tensor(size=(nf, nf), device=B.device)

    elif layout == "down":
        # Compute L = B.T @ B (Simplices x Simplices)
        # Same logic but on chunks of rows (faces)
        # L = sum over chunks of faces of (B_chunk.T @ B_chunk)
        return assemble_connectivity(B.t(), layout="up")

    return None


def get_routes_from_neighborhoods(neighborhoods):
    """Get the routes from the neighborhoods.

    Combination of src_rank, dst_rank. ex: [[0, 0], [1, 0], [1, 1], [1, 1], [2, 1]].

    Parameters
    ----------
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    list
        List of routes.
    """
    routes = []
    for neighborhood in neighborhoods:
        split = neighborhood.split("-")
        src_rank = int(split[-1])
        r = int(split[0]) if len(split) == 3 else 1
        route = (
            [src_rank, src_rank - r]
            if "down" in neighborhood
            else [src_rank, src_rank + r]
        )
        routes.append(route)
    return routes


def get_complex_connectivity(
    complex, max_rank, neighborhoods=None, signed=False
):
    """Get the connectivity matrices for the complex.

    Parameters
    ----------
    complex : toponetx.CellComplex or toponetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]:
            try:  #### from_sparse doesn't have rank and signed
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(
                        rank=rank_idx, signed=signed
                    )
                )
            except:  # noqa: E722
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    if neighborhoods is not None:
        connectivity = select_neighborhoods_of_interest(
            connectivity, neighborhoods
        )
    connectivity["shape"] = practical_shape
    return connectivity


def get_complex_connectivity_from_incidences(
    incidences, shape, max_rank, neighborhoods=None, signed=False
):
    """Compute connectivity matrices directly from sparse incidence matrices.

    This bypasses toponetx object construction by computing Laplacians,
    adjacency, and coadjacency matrices via PyTorch sparse algebra.

    Parameters
    ----------
    incidences : dict
        Dictionary mapping rank (int) to sparse PyTorch incidence matrices.
        E.g. {1: B1 (nodes x edges), 2: B2 (edges x triangles), ...}.
    shape : list
        List of counts per rank, e.g. [num_nodes, num_edges, num_triangles].
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, the incidence matrices carry orientation signs.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(

        np.pad(shape, (0, max(0, max_rank + 1 - len(shape))))
    )
    connectivity = {}

    for rank_idx in range(max_rank + 1):
        n_r = practical_shape[rank_idx]

        # --- 1. incidence_r (r-1 -> r) ---
        B_r = incidences.get(rank_idx)
        if B_r is not None:
            if not signed:
                connectivity[f"incidence_{rank_idx}"] = torch.sparse_coo_tensor(
                    B_r.indices(), B_r.values().abs(), B_r.size()
                ).coalesce()
            else:
                connectivity[f"incidence_{rank_idx}"] = B_r
        else:
            n_prev = practical_shape[rank_idx - 1] if rank_idx > 0 else 1
            connectivity[f"incidence_{rank_idx}"] = (
                generate_zero_sparse_connectivity(m=n_prev, n=n_r)
            )

        # --- 2. up_laplacian & adjacency (rank r) ---
        # Both derive from product B_{r+1} @ B_{r+1}.T
        B_rp1 = incidences.get(rank_idx + 1)
        if B_rp1 is not None:
            # For adjacency, TopoNetX uses |B| @ |B|.T when unsigned. 
            # If signed=False, B_rp1 is already absolute (handled in incidence_r+1 or manually here).
            if not signed:
                B_rp1_proc = torch.sparse_coo_tensor(B_rp1.indices(), B_rp1.values().abs(), B_rp1.size())
            else:
                B_rp1_proc = B_rp1

            # Core product for rank r "up" connectivity
            up_prod = assemble_connectivity(B_rp1_proc, layout="up")
            
            # Up Laplacian
            connectivity[f"up_laplacian_{rank_idx}"] = up_prod
            
            # Adjacency (off-diagonal entries of the product)
            idx = up_prod.indices()
            vals = up_prod.values()
            mask = idx[0] != idx[1]
            connectivity[f"adjacency_{rank_idx}"] = torch.sparse_coo_tensor(
                idx[:, mask], vals[mask].abs(), up_prod.size()
            ).coalesce()
        else:
            connectivity[f"up_laplacian_{rank_idx}"] = (
                generate_zero_sparse_connectivity(m=n_r, n=n_r)
            )
            connectivity[f"adjacency_{rank_idx}"] = (
                generate_zero_sparse_connectivity(m=n_r, n=n_r)
            )

        # --- 3. down_laplacian & coadjacency (rank r) ---
        # Both derive from product B_r.T @ B_r
        if B_r is not None and rank_idx >= 1:
            if not signed:
                B_r_proc = torch.sparse_coo_tensor(B_r.indices(), B_r.values().abs(), B_r.size())
            else:
                B_r_proc = B_r

            # Core product for rank r "down" connectivity
            down_prod = assemble_connectivity(B_r_proc, layout="down")
            
            # Down Laplacian
            connectivity[f"down_laplacian_{rank_idx}"] = down_prod
            
            # Coadjacency (off-diagonal entries of the product)
            idx = down_prod.indices()
            vals = down_prod.values()
            mask = idx[0] != idx[1]
            connectivity[f"coadjacency_{rank_idx}"] = torch.sparse_coo_tensor(
                idx[:, mask], vals[mask].abs(), down_prod.size()
            ).coalesce()
        else:
            connectivity[f"down_laplacian_{rank_idx}"] = (
                generate_zero_sparse_connectivity(m=n_r, n=n_r)
            )
            connectivity[f"coadjacency_{rank_idx}"] = (
                generate_zero_sparse_connectivity(m=n_r, n=n_r)
            )

        # --- 4. hodge_laplacian (rank r) ---
        connectivity[f"hodge_laplacian_{rank_idx}"] = (
            connectivity[f"down_laplacian_{rank_idx}"]
            + connectivity[f"up_laplacian_{rank_idx}"]
        ).coalesce()

    if neighborhoods is not None:
        connectivity = select_neighborhoods_of_interest(
            connectivity, neighborhoods
        )
    connectivity["shape"] = practical_shape
    return connectivity


def get_combinatorial_complex_connectivity(
    complex, max_rank, neighborhoods=None
):
    r"""Get the connectivity matrices for the Combinatorial Complex.

    Parameters
    ----------
    complex : topnetx.CombinatorialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]:
            try:
                if connectivity_info == "adjacency":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        from_sparse(
                            getattr(complex, f"{connectivity_info}_matrix")(
                                rank_idx, rank_idx + 1
                            )
                        )
                    )
                else:  # incidence
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        from_sparse(
                            getattr(complex, f"{connectivity_info}_matrix")(
                                rank_idx - 1, rank_idx
                            )
                        )
                    )
            except ValueError:
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
            except AttributeError:
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    if neighborhoods is not None:
        connectivity = select_neighborhoods_of_interest(
            connectivity, neighborhoods
        )
    connectivity["shape"] = practical_shape
    return connectivity


def select_neighborhoods_of_interest(connectivity, neighborhoods):
    """Select the neighborhoods of interest.

    Parameters
    ----------
    connectivity : dict
        Connectivity matrices generated by default.
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    dict
        Connectivity matrices of interest.
    """

    def generate_adjacency_from_laplacian(sparse_tensor):
        """Generate an adjacency matrix from a Laplacian matrix.

        Parameters
        ----------
        sparse_tensor : torch.sparse_coo_tensor
            Sparse tensor representing the Laplacian matrix.

        Returns
        -------
        torch.sparse_coo_tensor
            Sparse tensor representing the adjacency matrix.
        """
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()

        # Create a mask for non-diagonal elements
        mask = indices[0] != indices[1]

        # Filter indices and values based on the mask
        new_indices = indices[:, mask]
        new_values = values[mask]

        # Turn values to 1s
        new_values = new_values / new_values

        # Construct a new sparse tensor
        return torch.sparse_coo_tensor(
            new_indices, new_values, sparse_tensor.size()
        )

    useful_connectivity = {}
    for neighborhood in neighborhoods:
        src_rank = int(neighborhood.split("-")[-1])
        try:
            if (
                len(neighborhood.split("-")) == 2
                or neighborhood.split("-")[0] == "1"
            ):
                r = 1
                neighborhood_type = (
                    neighborhood.split("-")[0]
                    if neighborhood.split("-")[0] != "1"
                    else neighborhood.split("-")[1]
                )
                if "adjacency" in neighborhood_type:
                    useful_connectivity[neighborhood] = (
                        connectivity[f"adjacency_{src_rank}"]
                        if "up" in neighborhood_type
                        else connectivity[f"coadjacency_{src_rank}"]
                    )
                elif "laplacian" in neighborhood_type:
                    useful_connectivity[neighborhood] = connectivity[
                        f"{neighborhood_type}_{src_rank}"
                    ]
                elif "incidence" in neighborhood_type:
                    useful_connectivity[neighborhood] = (
                        connectivity[f"incidence_{src_rank + 1}"].T
                        if "up" in neighborhood_type
                        else connectivity[f"incidence_{src_rank}"]
                    )
            elif len(neighborhood.split("-")) == 3:
                r = int(neighborhood.split("-")[0])
                neighborhood_type = neighborhood.split("-")[1]
                if (
                    "adjacency" in neighborhood_type
                    or "laplacian" in neighborhood_type
                ):
                    direction, connectivity_type = neighborhood_type.split("_")
                    if direction == "up":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        # ORIGINAL: matrix = torch.sparse.mm(...)
                        matrix = fast_sparse_mm(
                            connectivity[f"incidence_{src_rank + 1}"],
                            connectivity[f"incidence_{src_rank + 2}"],
                        )
                        for idx in range(src_rank + 3, src_rank + r + 1):
                            matrix = fast_sparse_mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Multiply the resulting matrix by its transpose to get the laplacian matrix
                        matrix = fast_sparse_mm(matrix, matrix.T)
                        # Turn all values to 1s
                        matrix = torch.sparse_coo_tensor(
                            matrix.indices(),
                            matrix.values() / (matrix.values() + 1e-9), # Avoid div zero
                            matrix.size(),
                        )
                        # Generate the adjacency matrix from the laplacian if needed
                        useful_connectivity[neighborhood] = (
                            generate_adjacency_from_laplacian(matrix)
                            if "adjacency" in neighborhood_type
                            else matrix
                        )
                    elif direction == "down":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = fast_sparse_mm(
                            connectivity[f"incidence_{src_rank - r + 1}"],
                            connectivity[f"incidence_{src_rank - r + 2}"],
                        )
                        for idx in range(src_rank - r + 3, src_rank + 1):
                            matrix = fast_sparse_mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Multiply the resulting matrix by its transpose to get the laplacian matrix
                        matrix = fast_sparse_mm(matrix.T, matrix)
                        # Turn all values to 1s
                        matrix = torch.sparse_coo_tensor(
                            matrix.indices(),
                            matrix.values() / (matrix.values() + 1e-9),
                            matrix.size(),
                        )
                        # Generate the adjacency matrix from the laplacian if needed
                        useful_connectivity[neighborhood] = (
                            generate_adjacency_from_laplacian(matrix)
                            if "adjacency" in neighborhood_type
                            else matrix
                        )
                elif "incidence" in neighborhood_type:
                    direction, connectivity_type = neighborhood_type.split("_")
                    if direction == "up":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = fast_sparse_mm(
                            connectivity[f"incidence_{src_rank + 1}"],
                            connectivity[f"incidence_{src_rank + 2}"],
                        )
                        for idx in range(src_rank + 3, src_rank + r + 1):
                            matrix = fast_sparse_mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Turn all values to 1s and transpose the matrix
                        useful_connectivity[neighborhood] = (
                            torch.sparse_coo_tensor(
                                matrix.indices(),
                                matrix.values() / (matrix.values() + 1e-9),
                                matrix.size(),
                            ).T
                        )
                    elif direction == "down":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = fast_sparse_mm(
                            connectivity[f"incidence_{src_rank - r + 1}"],
                            connectivity[f"incidence_{src_rank - r + 2}"],
                        )
                        for idx in range(src_rank - r + 3, src_rank + 1):
                            matrix = fast_sparse_mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Turn all values to 1s
                        useful_connectivity[neighborhood] = (
                            torch.sparse_coo_tensor(
                                matrix.indices(),
                                matrix.values() / (matrix.values() + 1e-9),
                                matrix.size(),
                            )
                        )
            else:
                useful_connectivity[neighborhood] = connectivity[neighborhood]
        except:  # noqa: E722
            raise ValueError(f"Invalid neighborhood {neighborhood}")  # noqa: B904
    for key in connectivity:
        if "incidence" in key and "-" not in key:
            useful_connectivity[key] = connectivity[key]
    return useful_connectivity


def generate_zero_sparse_connectivity(m, n):
    """Generate a zero sparse connectivity matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    torch.sparse_coo_tensor
        Zero sparse connectivity matrix.
    """
    return torch.sparse_coo_tensor((m, n)).coalesce()


def load_cell_complex_dataset(cfg):
    r"""Load cell complex datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.
    """
    raise NotImplementedError


def load_simplicial_dataset(cfg):
    """Load simplicial datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Simplicial dataset.
    """
    raise NotImplementedError


def load_manual_graph():
    """Create a manual graph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual graph.
    """
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [5, 7],
        [2, 7],
        [0, 7],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )


def load_manual_graph_second_structure():
    """Create a manual graph for testing purposes with updated edges and node features.

    Returns
    -------
    torch_geometric.data.Data
        A simple graph data object.
    """
    # Define the vertices (12 vertices, based on the highest index in edges)
    vertices = [i for i in range(12)]
    y = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

    # Updated edges
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (3, 4),
        (3, 6),
        (3, 9),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 0),
        (5, 7),
        (5, 10),
        (5, 11),
        (6, 9),
        (7, 8),
        (8, 6),
        (10, 11),
    ]

    # Create a graph
    G = nx.Graph()

    # Add vertices and edges
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate updated features (example features for 12 nodes)
    x = (
        torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000, 200, 300, 400, 600])
        .unsqueeze(1)
        .float()
    )

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )
    return data


def ensure_serializable(obj):
    """Ensure that the object is serializable.

    Parameters
    ----------
    obj : object
        Object to ensure serializability.

    Returns
    -------
    object
        Object that is serializable.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    elif isinstance(obj, list | tuple):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, str | int | float | bool | type(None)):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        return dict(obj)
    else:
        return None


def make_hash(o):
    """Make a hash from a dictionary, list, tuple or set to any level, that contains only other hashable types.

    Parameters
    ----------
    o : dict, list, tuple, set
        Object to hash.

    Returns
    -------
    int
        Hash of the object.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # Convert the hex back to int and restrict it to the relevant int range
    return int(hash_as_hex, 16) % 4294967295


def load_manual_hypergraph():
    """Create a manual hypergraph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual hypergraph.
    """
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the hyperedges
    hyperedges = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
        [3, 4],
        [4, 5],
        [4, 7],
        [5, 6],
        [6, 7],
    ]

    # Generate feature from 0 to 7
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()
    labels = torch.tensor(y, dtype=torch.long)

    node_list = []
    edge_list = []

    for edge_idx, he in enumerate(hyperedges):
        cur_size = len(he)
        node_list += he
        edge_list += [edge_idx] * cur_size

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    incidence_hyperedges = torch.sparse_coo_tensor(
        edge_index,
        values=torch.ones(edge_index.shape[1]),
        size=(len(vertices), len(hyperedges)),
    )

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        incidence_hyperedges=incidence_hyperedges,
    )


def load_manual_pointcloud(pos_to_x: bool = False):
    """Create a manual pointcloud for testing purposes.

    Parameters
    ----------
    pos_to_x : bool, optional
        If True, the positions are used as features.

    Returns
    -------
    torch_geometric.data.Data
        Manual pointcloud.
    """
    # Define the positions
    pos = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1],
            [20, 0, 0],
            [20, 0, 1],
            [20, 1, 0],
            [20, 1, 1],
            [30, 0, 0],
        ]
    ).float()

    if pos_to_x:
        return torch_geometric.data.Data(
            x=pos, pos=pos, num_nodes=pos.size(0), num_features=pos.size(1)
        )

    return torch_geometric.data.Data(
        pos=pos, num_nodes=pos.size(0), num_features=0
    )


def load_manual_points():
    """Create a manual point cloud for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual point cloud.
    """
    pos = torch.tensor(
        [
            [1.0, 1.0],
            [7.0, 0.0],
            [4.0, 6.0],
            [9.0, 6.0],
            [0.0, 14.0],
            [2.0, 19.0],
            [9.0, 17.0],
        ],
        dtype=torch.float,
    )
    y = torch.randint(0, 2, (pos.shape[0],), dtype=torch.float)
    return torch_geometric.data.Data(x=pos, y=y, complex_dim=0)


def load_manual_simplicial_complex():
    """Create a manual simplicial complex for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual simplicial complex.
    """
    num_feats = 2
    one_cells = [i for i in range(5)]
    two_cells = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [0, 4], [2, 4]]
    three_cells = [[0, 1, 2], [1, 2, 3], [0, 2, 4]]
    incidence_1 = [
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ]
    incidence_2 = [
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]

    y = [1]

    return torch_geometric.data.Data(
        x_0=torch.rand(len(one_cells), num_feats),
        x_1=torch.rand(len(two_cells), num_feats),
        x_2=torch.rand(len(three_cells), num_feats),
        incidence_0=torch.zeros((1, 5)).to_sparse(),
        adjacency_1=torch.zeros((len(one_cells), len(one_cells))).to_sparse(),
        adjacency_2=torch.zeros((len(two_cells), len(two_cells))).to_sparse(),
        adjacency_0=torch.zeros((5, 5)).to_sparse(),
        incidence_1=torch.tensor(incidence_1).to_sparse(),
        incidence_2=torch.tensor(incidence_2).to_sparse(),
        num_nodes=len(one_cells),
        y=torch.tensor(y),
    )


def data2simplicial(data):
    """
    Convert a data dictionary into a SimplicialComplex object.

    Parameters
    ----------
    data : dict
        A dictionary containing at least 'incidence_0', 'adjacency_0', 'incidence_1',
        'incidence_2', and optionally 'incidence_3' tensors.

    Returns
    -------
    SimplicialComplex
        A SimplicialComplex object constructed from nodes, edges, triangles, and tetrahedrons.
    """
    sc = SimplicialComplex()

    # Nodes as single-element lists
    nodes = [[i] for i in range(data["incidence_0"].shape[1])]

    # Convert edges to a list of pairs
    edges = torch_geometric.utils.remove_self_loops(
        data["adjacency_0"].indices()
    )[0].T.tolist()

    # Detect triangles if incidence_1 and incidence_2 exist
    triangles = (
        find_triangles(data["incidence_1"], data["incidence_2"])
        if "incidence_1" in data and "incidence_2" in data
        else []
    )

    # Detect tetrahedrons if incidence_3 exists
    tetrahedrons = (
        find_tetrahedrons(
            data["incidence_1"], data["incidence_2"], data["incidence_3"]
        )
        if "incidence_3" in data
        else []
    )

    # Add simplices to the complex
    sc.add_simplices_from(nodes)
    sc.add_simplices_from(edges)
    sc.add_simplices_from(triangles)
    sc.add_simplices_from(tetrahedrons)

    return sc


def find_triangles(incidence_1, incidence_2):
    """
    Identify triangles in the simplicial complex based on incidence matrices.

    Parameters
    ----------
    incidence_1 : torch.Tensor
        Incidence matrix of edges.
    incidence_2 : torch.Tensor
        Incidence matrix of triangles.

    Returns
    -------
    list of list
        List of triangles, where each triangle is a list of three node indices.
    """
    triangles = (incidence_1 @ incidence_2).indices()
    unique_triangles = torch.unique(triangles[1])
    triangle_list = [
        [j.item() for j in triangles[0][torch.where(triangles[1] == i)[0]]]
        for i in unique_triangles
    ]
    return triangle_list


def find_tetrahedrons(incidence_1, incidence_2, incidence_3):
    """
    Identify tetrahedrons in the simplicial complex.

    Parameters
    ----------
    incidence_1 : torch.Tensor
        Incidence matrix of edges.
    incidence_2 : torch.Tensor
        Incidence matrix of triangles.
    incidence_3 : torch.Tensor
        Incidence matrix of tetrahedrons.

    Returns
    -------
    list of list
        List of tetrahedrons, where each is represented as a list of four node indices.
    """
    tetrahedrons = (incidence_1 @ incidence_2 @ incidence_3).indices()
    unique_tetrahedrons = torch.unique(tetrahedrons[1])
    tetrahedron_list = [
        [
            j.item()
            for j in tetrahedrons[0][torch.where(tetrahedrons[1] == i)[0]]
        ]
        for i in unique_tetrahedrons
    ]
    return tetrahedron_list


def zero_sparse(m, n, device=None, dtype=torch.float):
    """Generate a zero sparse tensor of given dimensions."""
    return torch.sparse_coo_tensor(size=(m, n), device=device, dtype=dtype).coalesce()

def binarize_sparse(A):
    """Binarize a sparse tensor by turning all non-zero values into 1.0."""
    if not A.is_sparse:
        return A
    A = A.coalesce()
    vals = torch.ones_like(A.values())
    return torch.sparse_coo_tensor(A.indices(), vals, A.size(), device=A.device).coalesce()

def abs_sparse(A):
    """Take the absolute value of all entries in a sparse tensor."""
    if not A.is_sparse:
        return A
    A = A.coalesce()
    vals = torch.abs(A.values())
    return torch.sparse_coo_tensor(A.indices(), vals, A.size(), device=A.device).coalesce()

def remove_diag_sparse(A, binarize=True):
    """Remove diagonal elements from a sparse tensor and optionally binarize."""
    if not A.is_sparse:
        return A
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    mask = indices[0] != indices[1]
    new_indices = indices[:, mask]
    new_values = values[mask]
    if binarize:
        new_values = torch.ones_like(new_values)
    return torch.sparse_coo_tensor(new_indices, new_values, A.size(), device=A.device).coalesce()

def safe_sparse_mm(A, B):
    """Safe sparse matrix multiplication."""
    A = A.coalesce()
    B = B.coalesce()
    if A._nnz() == 0 or B._nnz() == 0:
        return zero_sparse(A.size(0), B.size(1), device=A.device, dtype=A.dtype)
    return torch.sparse.mm(A, B)

def build_edge_cycle_adjacency_from_cycle_edges(cycle_edges, num_edges, device=None, dtype=torch.float, chunk_size=1000):
    """Return sparse COO equivalent to offdiag(B2 @ B2.T)."""
    if len(cycle_edges) == 0:
        return zero_sparse(num_edges, num_edges, device=device, dtype=dtype)
        
    all_rows, all_cols = [], []
    rows, cols = [], []
    
    for i, edges in enumerate(cycle_edges):
        if len(edges) < 2:
            continue
        e = torch.tensor(edges, dtype=torch.long, device=device)
        r = e.repeat_interleave(len(e))
        c = e.repeat(len(e))
        mask = r != c
        rows.append(r[mask])
        cols.append(c[mask])
        
        if chunk_size and (i + 1) % chunk_size == 0 and rows:
            all_rows.append(torch.cat(rows))
            all_cols.append(torch.cat(cols))
            rows, cols = [], []
            
    if rows:
        all_rows.append(torch.cat(rows))
        all_cols.append(torch.cat(cols))

    if not all_rows:
        return zero_sparse(num_edges, num_edges, device=device, dtype=dtype)

    row_t = torch.cat(all_rows)
    col_t = torch.cat(all_cols)
    val_t = torch.ones(len(row_t), dtype=dtype, device=device)
    
    adj = torch.sparse_coo_tensor(torch.stack([row_t, col_t]), val_t, size=(num_edges, num_edges), device=device).coalesce()
    return binarize_sparse(adj)

def build_node_cycle_adjacency_from_cycles(cycles, num_nodes, device=None, dtype=torch.float, chunk_size=1000):
    """Return sparse COO equivalent to offdiag((B1 @ B2) @ (B1 @ B2).T)."""
    if len(cycles) == 0:
        return zero_sparse(num_nodes, num_nodes, device=device, dtype=dtype)
        
    all_rows, all_cols = [], []
    rows, cols = [], []
    
    for i, nodes in enumerate(cycles):
        if len(nodes) < 2:
            continue
        n = torch.tensor(nodes, dtype=torch.long, device=device)
        r = n.repeat_interleave(len(n))
        c = n.repeat(len(n))
        mask = r != c
        rows.append(r[mask])
        cols.append(c[mask])
        
        if chunk_size and (i + 1) % chunk_size == 0 and rows:
            all_rows.append(torch.cat(rows))
            all_cols.append(torch.cat(cols))
            rows, cols = [], []
            
    if rows:
        all_rows.append(torch.cat(rows))
        all_cols.append(torch.cat(cols))

    if not all_rows:
        return zero_sparse(num_nodes, num_nodes, device=device, dtype=dtype)

    row_t = torch.cat(all_rows)
    col_t = torch.cat(all_cols)
    val_t = torch.ones(len(row_t), dtype=dtype, device=device)
    
    adj = torch.sparse_coo_tensor(torch.stack([row_t, col_t]), val_t, size=(num_nodes, num_nodes), device=device).coalesce()
    return binarize_sparse(adj)

def get_connectivity_from_incidences_selective(
    incidences,
    shape,
    max_rank,
    neighborhoods=None,
    signed=False,
    include_all_incidences=True,
    legacy_keys=True,
    cycles=None,
    cycle_edges=None,
    adjacency_strategy="pairs",
    required_keys=None,
    chunk_size=1000,
):
    """Memory-aware selective connectivity builder.
    
    If neighborhoods is None, it warns and returns only incidences to prevent OOM.
    """
    import warnings
    connectivity = {"shape": shape}
    
    if include_all_incidences:
        for r, inc in incidences.items():
            connectivity[f"incidence_{r}"] = inc

    if neighborhoods is None and required_keys is None:
        warnings.warn("No neighborhoods specified for selective builder. Returning only incidences to avoid OOM.")
        neighborhoods = []
        
    targets = set(neighborhoods or []) | set(required_keys or [])
    
    def get_inc(r):
        if r in incidences: return incidences[r]
        if r <= 0: return zero_sparse(0, shape[0] if len(shape) > 0 else 0)
        if r > len(shape) - 1: return zero_sparse(shape[-1] if len(shape) > 0 else 0, 0)
        return zero_sparse(shape[r-1], shape[r])
        
    for t in targets:
        t_normalized = t.replace("_", "-")
        
        if t_normalized == "up-adjacency-1" or t == "adjacency_1":
            if adjacency_strategy == "pairs" and cycle_edges is not None:
                ref_inc = get_inc(2)
                adj = build_edge_cycle_adjacency_from_cycle_edges(cycle_edges, shape[1], device=ref_inc.device, dtype=ref_inc.dtype, chunk_size=chunk_size)
            else:
                inc2 = get_inc(2)
                adj = remove_diag_sparse(safe_sparse_mm(inc2, inc2.T))
            connectivity[t] = adj
            if legacy_keys and t_normalized == "up-adjacency-1":
                connectivity["adjacency_1"] = adj
                connectivity["up_adjacency-1"] = adj

        elif t_normalized == "2-up-adjacency-0" or t == "2-up_adjacency-0":
            if adjacency_strategy == "pairs" and cycles is not None:
                ref_inc = get_inc(1)
                adj = build_node_cycle_adjacency_from_cycles(cycles, shape[0], device=ref_inc.device, dtype=ref_inc.dtype, chunk_size=chunk_size)
            else:
                inc1 = get_inc(1)
                inc2 = get_inc(2)
                B1B2 = safe_sparse_mm(inc1, inc2)
                adj = remove_diag_sparse(safe_sparse_mm(B1B2, B1B2.T))
            connectivity[t] = adj
            if legacy_keys:
                connectivity["2-up_adjacency-0"] = adj

        elif t_normalized == "up-laplacian-1" or t == "up_laplacian_1":
            inc2 = get_inc(2)
            uplap = safe_sparse_mm(inc2, inc2.T)
            if not signed:
                uplap = abs_sparse(uplap)
            connectivity[t] = uplap
            if legacy_keys:
                connectivity["up_laplacian_1"] = uplap
                connectivity["up_laplacian-1"] = uplap
                
        elif t_normalized == "down-laplacian-1" or t == "down_laplacian_1":
            inc1 = get_inc(1)
            downlap = safe_sparse_mm(inc1.T, inc1)
            if not signed:
                downlap = abs_sparse(downlap)
            connectivity[t] = downlap
            if legacy_keys:
                connectivity["down_laplacian_1"] = downlap
                connectivity["down_laplacian-1"] = downlap

        elif t_normalized == "up-adjacency-0" or t == "adjacency_0":
            inc1 = get_inc(1)
            adj = remove_diag_sparse(safe_sparse_mm(inc1, inc1.T))
            connectivity[t] = adj
            if legacy_keys:
                connectivity["adjacency_0"] = adj
                connectivity["up_adjacency-0"] = adj
                
        elif t_normalized == "down-adjacency-1" or t == "coadjacency_1":
            inc1 = get_inc(1)
            adj = remove_diag_sparse(safe_sparse_mm(inc1.T, inc1))
            connectivity[t] = adj
            if legacy_keys:
                connectivity["coadjacency_1"] = adj
                connectivity["down_adjacency-1"] = adj

        elif t_normalized == "up-incidence-0":
            connectivity[t] = get_inc(1).T
            if legacy_keys:
                connectivity["up_incidence-0"] = get_inc(1).T
                
        elif t_normalized == "down-incidence-2":
            connectivity[t] = get_inc(2)
            if legacy_keys:
                connectivity["down_incidence-2"] = get_inc(2)

        else:
            raise ValueError(f"Unsupported neighborhood: {t}")

    return connectivity
