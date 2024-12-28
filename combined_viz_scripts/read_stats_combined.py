import pandas as pd
import plotille

def plot_total_time_vs_variants(csv_path: str) -> None:
    # 1) Read in your CSV
    df = pd.read_csv(csv_path)

    # 2) Compute total variants = n_cpu_variants + n_gpu_variants
    df["n_variants"] = df["n_cpu_variants"] + df["n_gpu_variants"]

    # 3) For each unique simulation_name, plot total_time vs. n_variants
    for simulation in df['simulation_name'].unique():
        subset = df[df['simulation_name'] == simulation]

        print(f"Total Time vs Number of Variants for {simulation}")
        print(plotille.scatter(
            subset['n_variants'],
            subset['total_time'],
            width=60, 
            height=20
        ))
        print("X-axis: Variants, Y-axis: Total Time")
        print("\n")

if __name__ == "__main__":
    plot_total_time_vs_variants("performance_metrics_combined.csv")
