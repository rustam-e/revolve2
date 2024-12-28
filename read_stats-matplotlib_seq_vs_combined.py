import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(sequential_csv, combined_csv, output_prefix):
    # 1) --- Load & parse the "sequential" data ---
    df_seq = pd.read_csv(sequential_csv)
    # We assume the sequential CSV has:
    #  simulation_name, n_variants, cpu_time, gpu_time, ...
    # We'll create two subsets: one for CPU, one for GPU.

    # Subset for CPU
    df_seq_cpu = df_seq[["simulation_name", "n_variants", "cpu_time"]].copy()
    df_seq_cpu.rename(columns={"cpu_time": "time"}, inplace=True)
    df_seq_cpu["run_type"] = "Sequential CPU"

    # Subset for GPU
    df_seq_gpu = df_seq[["simulation_name", "n_variants", "gpu_time"]].copy()
    df_seq_gpu.rename(columns={"gpu_time": "time"}, inplace=True)
    df_seq_gpu["run_type"] = "Sequential GPU"

    # Combine them into one DataFrame
    df_seq_combined = pd.concat([df_seq_cpu, df_seq_gpu], ignore_index=True)

    # 2) --- Load & parse the "combined" data ---
    df_comb = pd.read_csv(combined_csv)
    # We assume the combined CSV has:
    #  simulation_name, total_time, n_cpu_variants, n_gpu_variants
    # We'll create a new column for n_variants = n_cpu_variants + n_gpu_variants
    # Then rename total_time -> time

    df_comb["n_variants"] = df_comb["n_cpu_variants"] + df_comb["n_gpu_variants"]
    df_comb.rename(columns={"total_time": "time"}, inplace=True)
    df_comb["run_type"] = "Combined"

    # We only need columns we plot:
    df_comb = df_comb[["simulation_name", "n_variants", "time", "run_type"]]

    # 3) --- Concatenate sequential (CPU/GPU) and combined ---
    df_all = pd.concat([df_seq_combined, df_comb], ignore_index=True)

    # 4) --- Convert columns to numeric just in case ---
    for col in ["n_variants", "time"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_all.dropna(subset=["simulation_name", "n_variants", "time"], inplace=True)

    # 5) --- For each simulation, plot overlay of three lines ---
    simulations = df_all["simulation_name"].unique()
    for sim in simulations:
        subset = df_all[df_all["simulation_name"] == sim]

        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=subset,
            x="n_variants",
            y="time",
            hue="run_type",
            marker="o",
            ci=None  # No confidence intervals
        )

        plt.title(f"{sim} - Overlay: Sequential CPU, Sequential GPU, Combined")
        plt.xlabel("Number of Variants")
        plt.ylabel("Time (seconds)")
        plt.legend(title="Run Type")
        plt.tight_layout()

        # Save
        filename = f"{output_prefix}_{sim}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved overlay for {sim} -> {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay sequential CPU/GPU vs. combined runs.")
    parser.add_argument("--sequential_csv", default="performance_metrics.csv",
                        help="Path to the sequential CSV file.")
    parser.add_argument("--combined_csv", default="performance_metrics_combined.csv",
                        help="Path to the combined CSV file.")
    parser.add_argument("--output_prefix", default="overlay",
                        help="Prefix for output plot filenames.")
    args = parser.parse_args()

    main(args.sequential_csv, args.combined_csv, args.output_prefix)
