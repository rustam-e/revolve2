
# Show specific plots:

# To show only the CPU Time vs Steps plot:
# python3 read_stats.py --cpu_vs_steps

# To show CPU and GPU Time vs Steps plots:
# python3 read_stats.py --cpu_vs_steps --gpu_vs_steps

# Show all plots:

# python3 read_stats.py --cpu_vs_steps --gpu_vs_steps --cpu_vs_variants --gpu_vs_variants


import argparse
import pandas as pd
import plotille

# Load the CSV file
df = pd.read_csv("performance_metrics.csv")

def plot_cpu_vs_steps(df):
    for simulation in df['simulation_name'].unique():
        subset = df[df['simulation_name'] == simulation]
        print(f"CPU Time vs Number of Steps for {simulation}")
        print(plotille.scatter(subset['n_steps'], subset['cpu_time'], width=60, height=20))
        print("X-axis: Steps, Y-axis: CPU Time")
        print("\n")

def plot_gpu_vs_steps(df):
    for simulation in df['simulation_name'].unique():
        subset = df[df['simulation_name'] == simulation]
        print(f"GPU Time vs Number of Steps for {simulation}")
        print(plotille.scatter(subset['n_steps'], subset['gpu_time'], width=60, height=20))
        print("X-axis: Steps, Y-axis: GPU Time")
        print("\n")

def plot_cpu_vs_variants(df):
    for simulation in df['simulation_name'].unique():
        subset = df[df['simulation_name'] == simulation]
        print(f"CPU Time vs Number of Variants for {simulation}")
        print(plotille.scatter(subset['n_variants'], subset['cpu_time'], width=60, height=20))
        print("X-axis: Variants, Y-axis: CPU Time")
        print("\n")

def plot_gpu_vs_variants(df):
    for simulation in df['simulation_name'].unique():
        subset = df[df['simulation_name'] == simulation]
        print(f"GPU Time vs Number of Variants for {simulation}")
        print(plotille.scatter(subset['n_variants'], subset['gpu_time'], width=60, height=20))
        print("X-axis: Variants, Y-axis: GPU Time")
        print("\n")

def main(show_cpu_vs_steps, show_gpu_vs_steps, show_cpu_vs_variants, show_gpu_vs_variants):
    if show_cpu_vs_steps:
        plot_cpu_vs_steps(df)
    if show_gpu_vs_steps:
        plot_gpu_vs_steps(df)
    if show_cpu_vs_variants:
        plot_cpu_vs_variants(df)
    if show_gpu_vs_variants:
        plot_gpu_vs_variants(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot performance data for CPU and GPU benchmarks.")
    parser.add_argument("--cpu_vs_steps", action="store_true", help="Show CPU Time vs Steps plot")
    parser.add_argument("--gpu_vs_steps", action="store_true", help="Show GPU Time vs Steps plot")
    parser.add_argument("--cpu_vs_variants", action="store_true", help="Show CPU Time vs Variants plot")
    parser.add_argument("--gpu_vs_variants", action="store_true", help="Show GPU Time vs Variants plot")
    
    args = parser.parse_args()

    main(
        show_cpu_vs_steps=args.cpu_vs_steps,
        show_gpu_vs_steps=args.gpu_vs_steps,
        show_cpu_vs_variants=args.cpu_vs_variants,
        show_gpu_vs_variants=args.gpu_vs_variants,
    )
