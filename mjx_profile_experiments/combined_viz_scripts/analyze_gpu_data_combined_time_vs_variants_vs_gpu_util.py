import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def main(csv_file="performance_metrics_combined.csv"):
    # 1. Load data
    df = pd.read_csv(csv_file)

    # 2. Because your CSV has "n_gpu_variants" and "n_cpu_variants",
    #    let's create "n_variants" from these columns.
    #    Make sure they actually exist in the CSV.
    df["n_variants"] = df["n_gpu_variants"] + df["n_cpu_variants"]

    # 3. Instead of "gpu_utilization", your CSV has "combined_gpu_utilization".
    #    We'll use that for coloring the points.
    #    Convert columns to numeric if needed:
    for col in ["n_variants", "total_time", "combined_gpu_utilization"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Drop rows if they’re missing critical info
    df.dropna(subset=["simulation_name", "n_variants", "total_time", "combined_gpu_utilization"], inplace=True)

    # 5. Create a Normalize object for GPU utilization
    norm = Normalize(
        vmin=df["combined_gpu_utilization"].min(), 
        vmax=df["combined_gpu_utilization"].max()
    )

    # 6. Get unique simulation names
    simulation_names = df["simulation_name"].unique()

    # 7. Iterate through each simulation and save individual plots
    for simulation_name in simulation_names:
        subset = df[df["simulation_name"] == simulation_name]

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            x=subset["n_variants"],
            y=subset["total_time"],
            c=norm(subset["combined_gpu_utilization"]),  # map utilization to color
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
        plt.ylabel("Total Time (s, log scale)")
        plt.title(f"Simulation: {simulation_name}")

        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label("GPU Utilization (%)")

        # Save the plot
        out_file = f"{simulation_name}_total_time_vs_variants_log.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved {out_file}")

    # 8. Create a FacetGrid for combined visualization
    g = sns.FacetGrid(
        df,
        col="simulation_name",
        col_wrap=3,
        height=4,
        sharex=False,
        sharey=False
    )

    # 9. Define a custom scatter function
    def scatter_with_color(data, **kwargs):
        """
        Plot total_time vs. n_variants, colored by combined_gpu_utilization.
        """
        plt.scatter(
            x=data["n_variants"],
            y=data["total_time"],
            c=norm(data["combined_gpu_utilization"]),
            cmap="plasma",
            edgecolor="k",
            alpha=0.7,
            s=60
        )

    # 10. Map our scatter function onto each facet
    g.map_dataframe(scatter_with_color)

    # 11. Use log scales (optional)
    for ax in g.axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Variants (log scale)")
        ax.set_ylabel("Total Time (s, log scale)")

    # 12. Single color bar for the entire figure
    cbar_ax = g.fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])  # needed so colorbar knows the correct range
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("GPU Utilization (%)")

    # 13. Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_file = "facet_total_time_vs_variants_log.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
