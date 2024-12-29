import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def main(csv_file="performance_metrics_combined.csv"):
    # 1. Load data
    df = pd.read_csv(csv_file)

    # 2. Create n_variants from CPU + GPU
    df["n_variants"] = df["n_gpu_variants"] + df["n_cpu_variants"]

    # 3. Compute the GPU variants percentage
    #    Handle zero-variants rows to avoid division by zero
    df["gpu_variants_percent"] = df.apply(
        lambda row: (row["n_gpu_variants"] / row["n_variants"] * 100) if row["n_variants"] != 0 else 0,
        axis=1
    )

    # 4. Convert columns to numeric if needed
    for col in ["n_variants", "total_time", "gpu_variants_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Drop rows that are missing the needed columns
    df.dropna(subset=["simulation_name", "n_variants", "total_time", "gpu_variants_percent"], inplace=True)

    # 6. Create a Normalize object for the GPU variants percentage
    norm = Normalize(
        vmin=df["gpu_variants_percent"].min(), 
        vmax=df["gpu_variants_percent"].max()
    )

    # 7. Get unique simulation names
    simulation_names = df["simulation_name"].unique()

    # 8. Iterate through each simulation and save individual plots
    for simulation_name in simulation_names:
        subset = df[df["simulation_name"] == simulation_name]

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            x=subset["n_variants"],
            y=subset["total_time"],
            c=norm(subset["gpu_variants_percent"]),
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
        cbar.set_label("GPU Variants (%)")

        # Save the plot
        out_file = f"{simulation_name}_total_time_vs_variants_gpu_percent.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved {out_file}")

    # 9. Create a FacetGrid for combined visualization
    g = sns.FacetGrid(
        df,
        col="simulation_name", 
        col_wrap=3,
        height=4,
        sharex=False,
        sharey=False
    )

    # 10. Define a custom plotting function
    def scatter_with_color(data, **kwargs):
        """
        Plot total_time vs. n_variants, colored by the percentage of GPU variants.
        """
        plt.scatter(
            x=data["n_variants"],
            y=data["total_time"],
            c=norm(data["gpu_variants_percent"]),
            cmap="plasma",
            edgecolor="k",
            alpha=0.7,
            s=60
        )

    # 11. Map the custom scatter function onto each facet
    g.map_dataframe(scatter_with_color)

    # 12. Log scales for each facet
    for ax in g.axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Variants (log scale)")
        ax.set_ylabel("Total Time (s, log scale)")

    # 13. Add a single color bar for the entire figure
    cbar_ax = g.fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("GPU Variants (%)")

    # 14. Adjust layout to accommodate color bar and save
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_file = "facet_total_time_vs_variants_gpu_percent.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
