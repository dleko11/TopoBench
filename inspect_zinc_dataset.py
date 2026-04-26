# inspect_ppi_dataset.py

from torch_geometric.datasets import ZINC
from pathlib import Path

# === 1. Define the dataset path ===
root = Path("datasets/graph/ZINC/ZINC")  # adjust if different

# === 2. Load the dataset splits ===
train_dataset = ZINC(root=str(root), split="train")
val_dataset   = ZINC(root=str(root), split="val")
test_dataset  = ZINC(root=str(root), split="test")

# === 3. Print dataset-level information ===
print("\n=== DATASET INFO ===")
print(f"Train graphs: {len(train_dataset)}")
print(f"Val graphs:   {len(val_dataset)}")
print(f"Test graphs:  {len(test_dataset)}")

# Print general attributes of one dataset
print("\nAttributes of train_dataset:")
print(dir(train_dataset))

# === 4. Inspect one graph (PyG Data object) ===
sample = train_dataset[0]
print("\n=== SAMPLE GRAPH ===")
print(sample)
print(f"\nNodes: {sample.num_nodes}")
print(f"Edges: {sample.num_edges}")
print(f"Node features shape: {sample.x.shape}")
print(f"Edge index shape: {sample.edge_index.shape}")
print(f"Label (y) shape: {sample.y.shape}")
print(f"Contains self-loops: {sample.has_self_loops()}")
print(f"Is undirected: {sample.is_undirected()}")

# Print all available attributes of the Data object
print("\nAll Data attributes and shapes:")
for key, value in sample.__dict__.items():
    if key != '_store':
        print(f"{key}: {type(value)}")
