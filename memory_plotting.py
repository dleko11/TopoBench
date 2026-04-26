import subprocess
import psutil
import time
import csv
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import ast
import tempfile

# Argument Parsing
parser = argparse.ArgumentParser(description="Monitor and plot memory usage of a test pipeline.")
parser.add_argument("--save_to", type=str, required=True, help='Prefix for the saved CSV and PNG files.')
parser.add_argument("--datasets", type=str, nargs=2, required=True, help='Two datasets to compare: InMemory then OnDisk.')
parser.add_argument("--models", type=str, nargs="+", required=True, help='One or more models to run.')
args = parser.parse_args()

print(f"Saving files with prefix: {args.save_to}")
print(f"Comparing datasets: {args.datasets}")
print(f"Running models: {args.models}")

SCRIPT_FILE_PATH = 'test/pipeline/test_pipeline_memory_checks.py'
OUTPUT_DIR = "Memory_checks"


def monitor_script(script_path, dataset_to_inject, model_to_inject, output_csv, interval=0.1):
    """
    Runs a modified, temporary copy of a script while monitoring its memory usage.
    """
    print(f"\n=== Running for model '{model_to_inject}' on dataset '{dataset_to_inject}' ===")

    project_root = os.getcwd()
    script_abspath = os.path.abspath(script_path)

    # Set up environment to ensure local modules can be imported
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    try:
        with open(script_abspath, "r") as f:
            original_content = f.read()
    except FileNotFoundError:
        print(f"ERROR: Script not found at {script_abspath}", file=sys.stderr)
        return None, -1

    # --- Safely modify the code using AST ---
    tree = ast.parse(original_content)

    class VariableReplacer(ast.NodeTransformer):
        def visit_Assign(self, node):
            # Ensure it's a simple assignment like 'VAR = ...'
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
                if target_name == 'DATASET':
                    # Replace the value with the new dataset string
                    node.value = ast.Constant(value=dataset_to_inject)
                elif target_name == 'MODELS':
                    # Replace the value with a new list containing the model string
                    node.value = ast.List(elts=[ast.Constant(value=model_to_inject)], ctx=ast.Load())
            return self.generic_visit(node)

    transformer = VariableReplacer()
    modified_tree = transformer.visit(tree)
    modified_code = ast.unparse(modified_tree)
    # --- End of AST modification ---

    # Get the directory of the original script
    original_script_dir = os.path.dirname(script_abspath)

    # Write the modified code to a temporary file IN THE SAME DIRECTORY
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, dir=original_script_dir) as temp_f:
        temp_f.write(modified_code)
        temp_script_path = temp_f.name

    print(f"Temporary config written to {temp_script_path}")

    # Launch the test process on the temporary script
    process = subprocess.Popen(["pytest", temp_script_path], cwd=project_root, env=env)
    ps_proc = psutil.Process(process.pid)
    memory_data = []
    start_time = time.time()

    try:
        while process.poll() is None:
            try:
                mem_info = ps_proc.memory_info()
                rss_mb = mem_info.rss / (1024 ** 2)  # bytes -> MB
                timestamp = time.time() - start_time
                memory_data.append((timestamp, rss_mb))
                time.sleep(interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break # Process finished before we could sample again
    except KeyboardInterrupt:
        print("Interrupted by user. Terminating child process.")
        process.terminate()

    process.wait()
    return_code = process.returncode
    total_runtime = time.time() - start_time

    print(f"=== Script finished in {total_runtime:.2f}s with return code {return_code} ===")

    # Write memory data to the specified CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'memory_MB'])
        writer.writerows(memory_data)
    print(f"Memory usage saved to {output_csv}")

    # Clean up the temporary file
    os.remove(temp_script_path)
    print(f"Temporary script {temp_script_path} removed.")

    return memory_data, return_code


def plot_normalized_memory(model, csv_files, labels, colors = ["blue", "red"]):
    """
    Plots memory usage from multiple CSV files on a single graph with normalized time.
    """
    plt.figure(figsize=(10, 6))

    for i, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found, skipping plot line: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        if df.empty:
            print(f"Warning: CSV file is empty, skipping plot line: {csv_file}")
            continue

        # Normalize time to a [0, 1] range
        total_duration = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
        if total_duration > 0:
            df['norm_time'] = (df['time_s'] - df['time_s'].iloc[0]) / total_duration
        else:
            df['norm_time'] = 0.0 # Avoid division by zero if runtime is too short

        plt.plot(df['norm_time'], df['memory_MB'], label=labels[i], color=colors[i])

    plt.xlabel("Normalized time")
    plt.ylabel("Memory usage [MB]")
    sanitized_model_name = model.replace('/', '_')
    plt.title(f"Memory usage comparison for model: {model}")
    plt.xlim(0, 1)
    plt.ylim(0, None)
    plt.legend(loc='best')
    plt.tight_layout()

    plot_filename = os.path.join(OUTPUT_DIR, f"{args.save_to}_{sanitized_model_name}.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"Saved plot: {plot_filename}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for model in args.models:
        csv_outputs_for_model = []
        return_codes = []

        for dataset in args.datasets:
            # Create sanitized, unique filenames for each run's output CSV
            sanitized_model_name = model.replace('/', '_')
            sanitized_dataset_name = dataset.replace('/', '_')
            output_csv = os.path.join(OUTPUT_DIR, f"{args.save_to}_{sanitized_model_name}_{sanitized_dataset_name}.csv")

            _, rc = monitor_script(SCRIPT_FILE_PATH, dataset, model, output_csv, interval=0.05)

            csv_outputs_for_model.append(output_csv)
            return_codes.append(rc)

        # Check for errors before plotting
        for i, rc in enumerate(return_codes):
            if rc != 0:
                print(f"WARNING: Run for model '{model}' on dataset '{args.datasets[i]}' exited with non-zero code {rc} (it may have crashed).", file=sys.stderr)

        # Plot the results for the current model
        plot_normalized_memory(model, csv_outputs_for_model, labels=['InMemoryDataset', 'OnDiskDataset'], colors = ["blue", "red"])