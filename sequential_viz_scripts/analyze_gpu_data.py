import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def main(csv_file="performance_metrics.csv"):
    # 1. Load data
    df = pd.read_csv(csv_file)

    # 2. Ensure numeric columns
    for col in ["n_variants", "gpu_time", "gpu_utilization"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["simulation_name", "n_variants", "gpu_time", "gpu_utilization"], inplace=True)

    # 3. Normalize object for GPU utilization (for continuous color scale)
    norm = Normalize(vmin=df["gpu_utilization"].min(), vmax=df["gpu_utilization"].max())

    # 4. Get unique simulation names
    simulation_names = df["simulation_name"].unique()

    # 5. Iterate through each simulation and save individual plots
    for simulation_name in simulation_names:
        subset = df[df["simulation_name"] == simulation_name]

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            x=subset["n_variants"],
            y=subset["gpu_time"],
            c=norm(subset["gpu_utilization"]),  # map utilization to color
            cmap="plasma",
            edgecolor="k",
            alpha=0.7,
            s=60
        )

        # Set log scale for x and y axes
        plt.xscale("log")
        plt.yscale("log")

        # Add labels and title
        plt.xlabel("Number of Variants (log scale)")
        plt.ylabel("GPU Time (s, log scale)")
        plt.title(f"Simulation: {simulation_name}")

        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label("GPU Utilization (%)")

        # Save the plot
        out_file = f"{simulation_name}_gpu_time_vs_variants_log.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
