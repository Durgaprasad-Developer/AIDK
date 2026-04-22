import subprocess
import sys
import os

print("🚀 Running benchmark + graph generation")

# Ensure PYTHONPATH is set so modules are found
env = {**os.environ, "PYTHONPATH": "."}

# Run benchmark
print("--- Step 1: Running Benchmark ---")
subprocess.run([sys.executable, "training/benchmark.py"], env=env, check=True)

# Generate graph
print("\n--- Step 2: Generating Graph ---")
subprocess.run([sys.executable, "training/plot_training_curve.py"], env=env, check=True)

print("\n✅ Graph + benchmark complete. Final Evidence Secure.")
