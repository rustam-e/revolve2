import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_cpu_gpu_by_variants(df):
    """
    Compare CPU vs GPU time across different n_variants, for each simulation.
    Produces a line plot (or scatter) with `n_variants` on the x-axis
    and time on the y-axis, hue = CPU or GPU.
    """
    # For each simulation, create a separate plot
    for sim in df['simulation_name'].unique():
        subset = df[df['simulation_name'] == sim].copy()
        
        # Only keep columns needed: n_variants, cpu_time, gpu_time
        subset = subset[['n_variants', 'cpu_time', 'gpu_time']]
        
        # Melt the CPU and GPU times into one column
        melted = subset.melt(
            id_vars='n_variants',
            value_vars=['cpu_time', 'gpu_time'],
            var_name='type',
            value_name='time'
        )
        
        # Create the plot
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=melted, x='n_variants', y='time', hue='type', marker='o')
        
        # If you prefer a scatter plot:
        # sns.scatterplot(data=melted, x='n_variants', y='time', hue='type')
        
        plt.title(f"CPU vs GPU Time across Variants ({sim})")
        plt.xlabel("Number of Variants")
        plt.ylabel("Time (seconds)")
        plt.legend(title="Type")
        plt.tight_layout()
        
        # Save the figure
        filename = f"cpu_gpu_vs_variants_{sim}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

def compare_cpu_gpu_by_steps(df):
    """
    Compare CPU vs GPU time across different n_steps, for each simulation.
    Produces a line plot (or scatter) with `n_steps` on the x-axis
    and time on the y-axis, hue = CPU or GPU.
    """
    for sim in df['simulation_name'].unique():
        subset = df[df['simulation_name'] == sim].copy()
        
        # Only keep columns needed: n_steps, cpu_time, gpu_time
        subset = subset[['n_steps', 'cpu_time', 'gpu_time']]
        
        # Melt the CPU and GPU times into one column
        melted = subset.melt(
            id_vars='n_steps',
            value_vars=['cpu_time', 'gpu_time'],
            var_name='type',
            value_name='time'
        )
        
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=melted, x='n_steps', y='time', hue='type', marker='o')
        
        plt.title(f"CPU vs GPU Time across Steps ({sim})")
        plt.xlabel("Number of Steps")
        plt.ylabel("Time (seconds)")
        plt.legend(title="Type")
        plt.tight_layout()
        
        filename = f"cpu_gpu_vs_steps_{sim}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

def main(csv_file, compare_variants, compare_steps):
    # 1. Read the CSV
    df = pd.read_csv(csv_file)

    # 2. (Optional) Remove repeated header rows if your CSV has them
    df = df[df['simulation_name'] != 'simulation_name']

    # 3. Convert columns to numeric if needed
    # (Uncomment if you have strings in numeric columns)
    # for col in ['n_variants', 'n_steps', 'cpu_time', 'gpu_time']:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    # df.dropna(subset=['n_variants', 'n_steps', 'cpu_time', 'gpu_time'], inplace=True)
    
    # 4. Perform comparisons
    if compare_variants:
        compare_cpu_gpu_by_variants(df)
    if compare_steps:
        compare_cpu_gpu_by_steps(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU for different simulations and variants/steps.")
    parser.add_argument("--csv_file", default="performance_metrics.csv",
                        help="Path to the CSV file containing performance metrics.")
    parser.add_argument("--compare_variants", action="store_true",
                        help="Compare CPU vs GPU time across n_variants for each simulation.")
    parser.add_argument("--compare_steps", action="store_true",
                        help="Compare CPU vs GPU time across n_steps for each simulation.")
    args = parser.parse_args()

    main(args.csv_file, args.compare_variants, args.compare_steps)
