import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(sequential_csv, combined_csv, output_prefix):
    # 1) --- Load & parse the "sequential" data ---
    df_seq = pd.read_csv(sequential_csv)
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
    df_comb_raw = pd.read_csv(combined_csv)
    df_comb_raw["n_variants"] = df_comb_raw["n_cpu_variants"] + df_comb_raw["n_gpu_variants"]

    df_comb = df_comb_raw[[
        "simulation_name", "total_time", "combined_cpu_time", "combined_gpu_time", "n_variants"
    ]].copy()
    df_comb.rename(columns={"total_time": "time"}, inplace=True)
    df_comb["run_type"] = "Combined"

    # 3) --- Add a row for "Naive Sum" = combined_cpu_time + combined_gpu_time
    df_naive = df_comb_raw[[
        "simulation_name", "n_variants", "combined_cpu_time", "combined_gpu_time"
    ]].copy()
    df_naive["time"] = df_naive["combined_cpu_time"] + df_naive["combined_gpu_time"]
    df_naive["run_type"] = "Naive Sum"
    df_naive = df_naive[["simulation_name", "n_variants", "time", "run_type"]]

    # Combine "real combined" and "naive sum" subsets
    df_comb_only = df_comb[["simulation_name", "n_variants", "time", "run_type"]]
    df_combined_plus_naive = pd.concat([df_comb_only, df_naive], ignore_index=True)

    # 5) --- Combine everything: sequential CPU/GPU, real combined, naive sum
    df_all = pd.concat([df_seq_combined, df_combined_plus_naive], ignore_index=True)

    # 6) --- Convert columns to numeric just in case ---
    for col in ["n_variants", "time"]:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_all.dropna(subset=["simulation_name", "n_variants", "time"], inplace=True)

    # 7) --- For each simulation, plot overlay of four lines ---
    simulations = df_all["simulation_name"].unique()
    for sim in simulations:
        subset = df_all[df_all["simulation_name"] == sim].copy()

        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=subset,
            x="n_variants",
            y="time",
            hue="run_type",
            marker="o",
            ci="sd"  # Display standard deviation as confidence intervals
        )

        plt.title(f"{sim} - Overlay: Seq CPU, Seq GPU, Combined, Naive Sum")
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
    parser = argparse.ArgumentParser(description="Overlay sequential CPU/GPU vs. combined runs plus naive sum.")
    parser.add_argument("--sequential_csv", default="performance_metrics.csv",
                        help="Path to the sequential CSV file.")
    parser.add_argument("--combined_csv", default="performance_metrics_combined.csv",
                        help="Path to the combined CSV file.")
    parser.add_argument("--output_prefix", default="overlay",
                        help="Prefix for output plot filenames.")
    args = parser.parse_args()

    main(args.sequential_csv, args.combined_csv, args.output_prefix)
