import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_total_time_vs_variants(csv_path: str) -> None:
    # 1) Read your CSV
    df = pd.read_csv(csv_path)

    # 2) Compute total variants
    df["n_variants"] = df["n_cpu_variants"] + df["n_gpu_variants"]

    # 3) For each unique simulation_name, plot total_time vs. n_variants
    for simulation in df["simulation_name"].unique():
        subset = df[df["simulation_name"] == simulation]

        plt.figure(figsize=(8, 6))

        # Use Seaborn for a line plot (with markers, no CI)
        sns.lineplot(
            data=subset,
            x="n_variants",
            y="total_time",
            marker='o',
            ci=None
        )

        plt.title(f"Total Time vs. Number of Variants for {simulation}")
        plt.xlabel("Number of Variants")
        plt.ylabel("Total Time (seconds)")

        # Build a filename and save the figure
        filename = f"{simulation}_time_vs_variants.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved {filename}")

if __name__ == "__main__":
    plot_total_time_vs_variants("performance_metrics_combined.csv")
