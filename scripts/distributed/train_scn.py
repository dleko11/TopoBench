"""Train SCN full-graph or halo-distributed under the paper's full-graph protocol.

Both modes share the model, initial weights, split, optimizer, schedule and
early stopping; they differ only in how the forward pass is computed.

Full graph (one GPU, TopoBench clique lifting and real modules):
    python scripts/distributed/train_scn.py --mode full --dataset cora_full

Halo-distributed (one process per GPU):
    torchrun --nproc_per_node 4 scripts/distributed/train_scn.py \
        --mode dist --dataset cora_full

Equivalence check (float64, dropout off; rank 0 also runs the full graph and
compares logits, gradients and the loss of every training step):
    torchrun --nproc_per_node 4 scripts/distributed/train_scn.py \
        --mode check --dataset cora_full
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.datasets import (
    Coauthor,
    CoraFull,
    HeterophilousGraphDataset,
    Yelp,
)
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_auroc,
    multilabel_f1_score,
)

from topobench.data.utils.split_utils import random_splitting
from topobench.distributed.encoder_first import EncoderFirstSCN
from topobench.distributed.halo import (
    build_halo_plan,
    canonical_edges,
    partition_nodes,
    vertex_weights,
)
from topobench.distributed.runtime import (
    DistributedSCN,
    all_reduce_gradients,
    build_local_plan,
    gather_rows,
)
from topobench.distributed.scn import (
    SCNNodeClassifier,
    build_reference_batch,
    cast_batch,
)
from topobench.distributed.stream import StreamedSCN

# Full-graph SCN hyperparameters selected in the paper (Table: optuna_hparams).
DATASETS = {
    "cora_full": dict(
        train_prop=0.7, metric="accuracy", lr=1e-3, wd=1e-3, hidden=128
    ),
    "amazon_ratings": dict(
        train_prop=0.5, metric="accuracy", lr=1e-2, wd=0.0, hidden=64
    ),
    "questions": dict(
        train_prop=0.5, metric="auroc", lr=1e-2, wd=1e-4, hidden=128
    ),
    # No full-graph hyperparameters exist for Physics (full graph did not fit);
    # these are the Cluster-TNN-selected SCN values and its 70% split.
    "coauthor_physics": dict(
        train_prop=0.7, metric="accuracy", lr=1e-2, wd=0.0, hidden=32
    ),
    # Inductive GraphSAINT split; SCN values from the Cluster-TNN Optuna
    # search (optuna_inductive_partitioning_yelp_split_fixed_inductive).
    "yelp": dict(
        split="fixed",
        inductive=True,
        multilabel=True,
        metric="f1_micro",
        lr=1e-3,
        wd=0.0,
        hidden=128,
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode",
        choices=["full", "dist", "check", "stream", "first"],
        required=True,
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--proj-dropout", type=float, default=0.3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--node-weight", choices=["none", "cells"], default="cells"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32"
    )
    parser.add_argument(
        "--feature-dim", type=int, default=0, help="0 keeps all features."
    )
    parser.add_argument("--check-steps", type=int, default=20)
    parser.add_argument(
        "--encoder-first",
        action="store_true",
        help="Encode inputs from node features in --mode dist/check.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=6,
        help="Parts on one GPU in --mode stream or first (1 = whole graph).",
    )
    parser.add_argument("--out", default="logs/distributed/train")
    return parser.parse_args()


def load_dataset(name: str, data_dir: str, feature_dim: int):
    """Load a dataset's features, symmetric canonical edges and labels.

    Parameters
    ----------
    name : str
        Dataset name.
    data_dir : str
        Root data directory.
    feature_dim : int
        Keep the most frequent feature columns, or 0 for all.

    Returns
    -------
    tuple
        Features, symmetric edge index, labels, and the fixed split code
        (0/1/2) when the dataset ships one, else ``None``.
    """
    fixed = None
    if name == "cora_full":
        data = CoraFull(root=f"{data_dir}/graph/cocitation/CoraFull")[0]
    elif name == "coauthor_physics":
        data = Coauthor(root=f"{data_dir}/graph/coauthor", name="Physics")[0]
    elif name == "yelp":
        data = Yelp(root=f"{data_dir}/graph/GraphSAINT/Yelp")[0]
        fixed = torch.full((data.num_nodes,), -1, dtype=torch.long)
        for value, key in enumerate(("train_mask", "val_mask", "test_mask")):
            fixed[data[key]] = value
    else:
        data = HeterophilousGraphDataset(
            root=f"{data_dir}/graph/heterophilic", name=name
        )[0]
    x = data.x.float()
    if feature_dim:
        columns = torch.argsort((x != 0).sum(0), descending=True)[:feature_dim]
        x = x[:, columns.sort().values]
    edges = canonical_edges(data.edge_index, data.num_nodes)
    labels = (
        data.y.float() if DATASETS[name].get("multilabel") else data.y.long()
    )
    return x, torch.cat([edges, edges.flip(0)], dim=1), labels, fixed


def split_masks(
    labels, name: str, data_dir: str, data_seed: int, train_prop: float
):
    """Return TopoBench's random split as a node-level code (0/1/2).

    Parameters
    ----------
    labels : torch.Tensor
        Node labels.
    name : str
        Dataset name, used for the split directory.
    data_dir : str
        Root data directory.
    data_seed : int
        Split fold, as ``dataset.split_params.data_seed``.
    train_prop : float
        Training proportion.

    Returns
    -------
    torch.Tensor
        0 for train, 1 for validation, 2 for test.
    """
    split = random_splitting(
        labels,
        {
            "data_seed": data_seed,
            "data_split_dir": f"{data_dir}/data_splits/{name}",
            "train_prop": train_prop,
        },
    )
    code = torch.full((labels.size(0),), -1, dtype=torch.long)
    for value, key in enumerate(("train", "valid", "test")):
        code[torch.as_tensor(split[key])] = value
    return code


def build_model(
    args, in_channels: int, num_classes: int, dtype
) -> SCNNodeClassifier:
    """Build the SCN with identical initial weights in every mode and process.

    Returns
    -------
    SCNNodeClassifier
        Model in the requested dtype.
    """
    torch.manual_seed(args.seed)
    model = SCNNodeClassifier(
        in_channels,
        DATASETS[args.dataset]["hidden"],
        num_classes,
        args.n_layers,
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.proj_dropout
    return model.to(dtype)


def metric_value(name: str, logits, labels, num_classes: int) -> float:
    """Compute the paper's metric (TopoBench evaluator settings).

    Returns
    -------
    float
        Accuracy, or macro AUROC for Questions.
    """
    logits, labels = logits.float().cpu(), labels.cpu()
    if name == "f1_micro":  # TopoBench: multilabel F1 on sigmoid outputs
        return float(
            multilabel_f1_score(
                logits.sigmoid(),
                labels.long(),
                num_labels=num_classes,
                average="micro",
            )
        )
    if name == "auroc":
        return float(
            multiclass_auroc(logits, labels, num_classes, average="macro")
        )
    return float(
        multiclass_accuracy(logits, labels, num_classes, average="micro")
    )


class FullGraph:
    """Full-graph forward: TopoBench's lifting and the real SCN modules."""

    def __init__(self, model, x, edge_index, labels, code, device, dtype):
        start = time.perf_counter()
        self.batch = cast_batch(
            build_reference_batch(x, edge_index, labels), dtype
        ).to(device)
        self.lift_seconds = time.perf_counter() - start
        self.timings = {"lift_seconds": self.lift_seconds}
        self.model = model
        self.labels = labels.to(device)
        self.code = code.to(device)
        self.counts = [int(self.batch[f"x_{r}"].size(0)) for r in range(3)]

    def __call__(self):
        # Shallow copy: the encoder swaps in new tensors, the inputs stay intact
        # and are not duplicated on the GPU.
        return self.model(copy.copy(self.batch))


class Distributed:
    """This process's share of the halo-distributed forward."""

    def __init__(
        self,
        model,
        x,
        edge_index,
        labels,
        code,
        part,
        device,
        dtype,
        encoder_first=False,
    ):
        num_nodes = x.size(0)
        edges = canonical_edges(edge_index, num_nodes)
        self.plan = build_local_plan(edges, part, num_nodes, device, dtype)
        start = time.perf_counter()
        self.forward = DistributedSCN(
            model,
            self.plan,
            x,
            num_nodes,
            device,
            dtype,
            encoder_first=encoder_first,
        )
        key = (
            "encoder_setup_seconds"
            if encoder_first
            else "feature_lift_seconds"
        )
        self.plan.timings[key] = time.perf_counter() - start
        self.setup_bytes_sent = sum(self.plan.traffic.sent.values())
        self.plan.traffic.reset()
        self.timings = self.plan.timings
        core = self.plan.shard.core
        self.nodes = core.to(device)
        self.labels = labels[core].to(device)
        self.code = code[core].to(device)
        self.counts = [cells.size(0) for cells in self.plan.shard.owned]

    def __call__(self):
        return self.forward()


class OneGPU:
    """One GPU: every part lifted in succession, then either the input
    encoder streamed by part (``stream``) or evaluated from node features
    (``first``, encoder-first)."""

    def __init__(
        self,
        model,
        x,
        edge_index,
        labels,
        code,
        parts,
        weight,
        device,
        dtype,
        kind="stream",
    ):
        num_nodes = x.size(0)
        self.timings = {}
        start = time.perf_counter()
        weights = (
            None
            if weight == "none"
            else vertex_weights(edge_index, num_nodes, "cells")
        )
        part = partition_nodes(
            edge_index, num_nodes, parts, "metis", node_weight=weights
        )
        self.timings["partition_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        plan = build_halo_plan(edge_index, num_nodes, part)
        self.timings["lift_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        if kind == "first":
            self.forward = EncoderFirstSCN(model, plan, x, device, dtype)
            self.host_feature_bytes = 0
            self.timings["encoder_setup_seconds"] = time.perf_counter() - start
        else:
            self.forward = StreamedSCN(model, plan, x, device, dtype)
            self.host_feature_bytes = sum(
                f.numel() * f.element_size() for f in self.forward.features
            )
            self.timings["feature_lift_seconds"] = time.perf_counter() - start
        nodes = plan.cells(0)[:, 0]
        self.nodes = nodes.to(device)
        self.labels = labels[nodes].to(device)
        self.code = code[nodes].to(device)
        self.counts = [plan.num_cells(r) for r in range(3)]

    def __call__(self):
        return self.forward()


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    """Sum a tensor across processes when distributed.

    Returns
    -------
    torch.Tensor
        Summed value.
    """
    if distributed:
        value = value.clone()
        dist.all_reduce(value)
    return value


def evaluate(
    runner, distributed: bool, metric: str, num_classes: int
) -> tuple:
    """Return validation and test metrics.

    Returns
    -------
    tuple[float, float]
        Validation and test metric.
    """
    with torch.no_grad():
        logits = runner()["logits"]
    out = []
    for split in (1, 2):
        mask = runner.code == split
        chunk_logits, chunk_labels = logits[mask], runner.labels[mask]
        if distributed:
            chunk_logits, chunk_labels = (
                gather_rows(chunk_logits),
                gather_rows(chunk_labels),
            )
        out.append(
            metric_value(metric, chunk_logits, chunk_labels, num_classes)
        )
    return tuple(out)


def supervised_loss(logits, labels, mask, count, multilabel) -> torch.Tensor:
    """Mean training loss over all processes' supervised rows.

    Cross-entropy per node, or binary cross-entropy per node and label
    (TopoBench's ``BCE`` for multilabel classification), summed locally and
    divided by the global count, so summed gradients are the full-graph ones.

    Returns
    -------
    torch.Tensor
        This process's share of the global mean loss.
    """
    if multilabel:
        total = F.binary_cross_entropy_with_logits(
            logits[mask], labels[mask], reduction="sum"
        )
        return total / (count * labels.size(1))
    return F.cross_entropy(logits[mask], labels[mask], reduction="sum") / count


def train(
    args, runner, model, num_classes, distributed, device, eval_runner=None
) -> dict:
    """Run the paper's full-graph training protocol.

    For inductive datasets ``runner`` covers the training graph and
    ``eval_runner`` the full graph used for validation and test.

    Returns
    -------
    dict
        History, best checkpoint metrics and timings.
    """
    spec = DATASETS[args.dataset]
    eval_runner = eval_runner or runner
    multilabel = bool(spec.get("multilabel"))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=spec["lr"], weight_decay=spec["wd"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5
    )
    train_mask = runner.code == 0
    train_count = reduce_sum(train_mask.sum(), distributed).item()
    history, checks = [], []
    best = {"val": float("-inf")}
    bad_checks = 0
    wall_start = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = runner()["logits"]
        loss = supervised_loss(
            logits, runner.labels, train_mask, train_count, multilabel
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if distributed:
            all_reduce_gradients(model)
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        history.append(
            {
                "epoch": epoch,
                "loss": reduce_sum(loss.detach(), distributed).item(),
                "seconds": seconds,
            }
        )
        if epoch % args.val_every:
            continue
        model.eval()
        eval_start = time.perf_counter()
        val, test = evaluate(
            eval_runner, distributed, spec["metric"], num_classes
        )
        eval_seconds = time.perf_counter() - eval_start
        elapsed = time.perf_counter() - wall_start
        checks.append(
            {
                "epoch": epoch,
                "val": val,
                "test": test,
                "eval_seconds": eval_seconds,
                "elapsed_seconds": elapsed,
            }
        )
        if val > best["val"]:
            best = {
                "epoch": epoch,
                "val": val,
                "test": test,
                "elapsed_seconds": elapsed,
            }
            bad_checks = 0
        else:
            bad_checks += 1
            if bad_checks >= args.patience:
                break
    return {
        "history": history,
        "checks": checks,
        "best": best,
        "train_wall_seconds": time.perf_counter() - wall_start,
    }


def run_check(
    args, x, edge_index, labels, code, part, num_classes, device
) -> dict:
    """Compare the distributed run with the full graph step by step (rank 0).

    Returns
    -------
    dict
        Max differences of logits, gradients and per-step losses.
    """
    dtype = torch.float64
    args.proj_dropout = 0.0
    # Both arms lift features in float32 (as ProjectionSum does) but sum in a
    # different order. On a 1/256 grid those sums are exact in any order, so
    # the check isolates the distributed computation from input rounding.
    x = torch.round(x * 256) / 256
    me = dist.get_rank()
    model = build_model(args, x.size(1), num_classes, dtype).to(device)
    runner = Distributed(
        model,
        x,
        edge_index,
        labels,
        code,
        part,
        device,
        dtype,
        encoder_first=args.encoder_first,
    )
    reference = None
    if me == 0:
        full_model = build_model(args, x.size(1), num_classes, dtype).to(
            device
        )
        reference = FullGraph(
            full_model, x, edge_index, labels, code, device, dtype
        )
        full_opt = torch.optim.Adam(
            full_model.parameters(), lr=DATASETS[args.dataset]["lr"]
        )
        full_train = reference.code == 0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=DATASETS[args.dataset]["lr"]
    )
    train_mask = runner.code == 0
    train_count = reduce_sum(train_mask.sum(), True).item()
    result = {"steps": []}
    for step in range(args.check_steps):
        logits = runner()["logits"]
        loss = (
            F.cross_entropy(
                logits[train_mask], runner.labels[train_mask], reduction="sum"
            )
            / train_count
        )
        optimizer.zero_grad()
        loss.backward()
        all_reduce_gradients(model)
        nodes, node_logits = (
            gather_rows(runner.nodes),
            gather_rows(logits.detach()),
        )
        dist_loss = reduce_sum(loss.detach(), True).item()
        if me == 0:
            full_logits = reference()["logits"]
            full_loss = F.cross_entropy(
                full_logits[full_train], reference.labels[full_train]
            )
            full_opt.zero_grad()
            full_loss.backward()
            row = {
                "step": step,
                "loss_full": full_loss.item(),
                "loss_dist": dist_loss,
                "max_logit_diff": float(
                    (full_logits.detach()[nodes] - node_logits).abs().max()
                ),
                "max_grad_diff": max(
                    float((a.grad - b.grad).abs().max())
                    for a, b in zip(
                        full_model.parameters(),
                        model.parameters(),
                        strict=True,
                    )
                ),
            }
            result["steps"].append(row)
            print(row, flush=True)
            full_opt.step()
        optimizer.step()
    return result


def training_subgraph(x, edge_index, labels, code):
    """Inductive training graph: the subgraph induced on training nodes.

    Matches TopoBench's ``load_inductive_node_splits``.

    Returns
    -------
    tuple
        Features, edges, labels and an all-training code, relabelled.
    """
    nodes = (code == 0).nonzero().flatten()
    relabel = torch.full((code.numel(),), -1, dtype=torch.long)
    relabel[nodes] = torch.arange(nodes.numel())
    keep = (relabel[edge_index[0]] >= 0) & (relabel[edge_index[1]] >= 0)
    edges = relabel[edge_index[:, keep]]
    return (
        x[nodes],
        edges,
        labels[nodes],
        torch.zeros(nodes.numel(), dtype=torch.long),
    )


def shared_partition(edge_index, num_nodes, world, weight, me, device):
    """Partition on rank 0 with cell-weighted METIS and broadcast it.

    Returns
    -------
    torch.Tensor
        Worker id of every node, on the CPU.
    """
    part = torch.empty(num_nodes, dtype=torch.long, device=device)
    if me == 0:
        weights = (
            None
            if weight == "none"
            else vertex_weights(edge_index, num_nodes, "cells")
        )
        part.copy_(
            partition_nodes(
                edge_index, num_nodes, world, "metis", node_weight=weights
            )
        )
    dist.broadcast(part, src=0)
    return part.cpu()


def make_runner(args, model, graph, part, device, dtype, distributed):
    """Build the forward runner for one graph in the requested mode.

    Returns
    -------
    object
        Runner exposing ``__call__``, ``labels``, ``code``, ``counts``.
    """
    x, edge_index, labels, code = graph
    if distributed:
        return Distributed(
            model,
            x,
            edge_index,
            labels,
            code,
            part,
            device,
            dtype,
            encoder_first=args.encoder_first,
        )
    if args.mode in ("stream", "first"):
        return OneGPU(
            model,
            x,
            edge_index,
            labels,
            code,
            args.parts,
            args.node_weight,
            device,
            dtype,
            kind=args.mode,
        )
    return FullGraph(model, x, edge_index, labels, code, device, dtype)


def main() -> None:
    """Entry point."""
    args = parse_args()
    distributed = args.mode in ("dist", "check")
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        me, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        me, world, local = 0, 1, 0
    device = torch.device(
        f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    spec = DATASETS[args.dataset]

    if distributed and me != 0:
        dist.barrier()  # let rank 0 download, process and write split files
    x, edge_index, labels, fixed = load_dataset(
        args.dataset, args.data_dir, args.feature_dim
    )
    if spec.get("split") == "fixed":
        code = fixed
    else:
        code = split_masks(
            labels,
            args.dataset,
            args.data_dir,
            args.data_seed,
            spec["train_prop"],
        )
    if distributed and me == 0:
        dist.barrier()
    num_classes = (
        labels.size(1) if spec.get("multilabel") else int(labels.max()) + 1
    )
    graphs = {"eval": (x, edge_index, labels, code)}
    if spec.get("inductive"):
        graphs["train"] = training_subgraph(x, edge_index, labels, code)

    parts = {}
    partition_seconds = 0.0
    if distributed:
        start = time.perf_counter()
        for key, (gx, gedges, _, _) in graphs.items():
            parts[key] = shared_partition(
                gedges, gx.size(0), world, args.node_weight, me, device
            )
        partition_seconds = time.perf_counter() - start
    part = parts.get("eval")

    if args.mode == "check":
        result = run_check(
            args, x, edge_index, labels, code, part, num_classes, device
        )
        report = {"mode": "check", "world": world, **result}
    else:
        model = build_model(args, x.size(1), num_classes, dtype).to(device)
        torch.manual_seed(args.seed + 1000 * me)  # dropout stream per process
        setup_start = time.perf_counter()
        setup_seconds, setup_bytes = 0.0, 0
        phase = "setup"
        try:
            runners = {
                key: make_runner(
                    args,
                    model,
                    graph,
                    parts.get(key),
                    device,
                    dtype,
                    distributed,
                )
                for key, graph in graphs.items()
            }
            runner = runners.get("train", runners["eval"])
            eval_runner = runners["eval"]
            setup_seconds = time.perf_counter() - setup_start
            setup_bytes = (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            phase = "train"
            outcome = train(
                args,
                runner,
                model,
                num_classes,
                distributed,
                device,
                eval_runner=eval_runner,
            )
        except torch.cuda.OutOfMemoryError:
            if distributed:
                raise  # torchrun tears the group down; the log records it
            report = {
                "mode": args.mode,
                "world": world,
                "oom": True,
                "oom_phase": phase,
                "setup_seconds": setup_seconds
                or time.perf_counter() - setup_start,
                "setup_peak_gpu_bytes": setup_bytes,
                "peak_before_oom_bytes": torch.cuda.max_memory_allocated(
                    device
                ),
                "gpu_total_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
                "args": vars(args),
            }
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            label = args.parts if args.mode in ("stream", "first") else world
            name = f"{args.dataset}_{args.mode}_P{label}_seed{args.data_seed}_{args.dtype}.json"
            (out / name).write_text(json.dumps(report, indent=1))
            print(f"OOM during {phase}; wrote {out / name}", flush=True)
            return
        train_peak = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        )
        traffic = runner.plan.traffic.sent if distributed else {}
        local = {
            "rank": me,
            "cells": runner.counts,
            "eval_cells": eval_runner.counts,
            "eval_timings": eval_runner.timings,
            "setup_seconds": setup_seconds,
            "setup_peak_gpu_bytes": setup_bytes,
            "train_peak_gpu_bytes": train_peak,
            "cpu_max_rss_kb": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "bytes_sent_forward_total": sum(traffic.values()),
            "bytes_sent_setup": runner.setup_bytes_sent if distributed else 0,
            "timings": runner.timings,
            "host_feature_bytes": getattr(runner, "host_feature_bytes", 0),
        }
        if distributed:
            gathered = [None] * world
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        epochs = outcome["history"]
        report = {
            "mode": args.mode,
            "world": world,
            "partition_seconds": partition_seconds,
            "num_forward_calls": len(epochs) + len(outcome["checks"]),
            "processes": gathered,
            **outcome,
            "mean_epoch_seconds": sum(e["seconds"] for e in epochs[1:])
            / max(len(epochs) - 1, 1),
        }
    if me == 0:
        report["args"] = vars(args)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        parts = args.parts if args.mode in ("stream", "first") else world
        name = f"{args.dataset}_{args.mode}_P{parts}_seed{args.data_seed}_{args.dtype}.json"
        (out / name).write_text(json.dumps(report, indent=1))
        best = report.get("best")
        print(
            f"wrote {out / name}" + (f" best={best}" if best else ""),
            flush=True,
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
