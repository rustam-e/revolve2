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

    # 4. Create a FacetGrid with one subplot per simulation
    g = sns.FacetGrid(
        df,
        col="simulation_name",    # separate subplot per unique simulation
        col_wrap=3,              # how many subplots per row
        height=4,                # size (in inches) for each subplot
        sharex=False,            # allow separate scaling on x
        sharey=False             # and y for each plot if desired
    )

    # 5. Define a custom plotting function to map per subset
    def scatter_with_color(data, **kwargs):
        """
        Plots a scatter of GPU time vs. n_variants, 
        color-coded by GPU utilization on a continuous scale.
        """
        plt.scatter(
            x=data["n_variants"],
            y=data["gpu_time"],
            c=norm(data["gpu_utilization"]),  # map utilization to color
            cmap="plasma",
            edgecolor="k",
            alpha=0.7,
            s=60
        )

    # 6. Map the custom scatter function onto each facet
    g.map_dataframe(scatter_with_color)

    # 7. Set log scale for x and y axes in each facet
    for ax in g.axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Variants (log scale)")
        ax.set_ylabel("GPU Time (s, log scale)")

    # 8. Add a single color bar for the entire figure
    #    We'll manually create a ScalarMappable for the color bar
    #    and place it off to the right.
    cbar_ax = g.fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
    sm.set_array([])  # Hack: set_array([]) so colorbar finds the correct range
    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("GPU Utilization (%)")

    # 9. Adjust the layout so the subplots & color bar fit nicely
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space on the right for color bar

    # 10. Save the figure
    out_file = "facet_gpu_time_vs_variants_log.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
