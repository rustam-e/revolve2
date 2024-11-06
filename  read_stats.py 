import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("performance_metrics.csv")

# Set display option to show all rows if needed
pd.set_option('display.max_rows', None)

# Display all rows
print(df)

# Plot CPU and GPU times by number of steps and variants
def plot_time_vs_steps_variants(df):
    # CPU Time vs Steps
    plt.figure()
    for simulation in df['simulation'].unique():
        subset = df[df['simulation'] == simulation]
        plt.scatter(subset['n_steps'], subset['cpu_time'], label=f'CPU - {simulation}', alpha=0.6)
    plt.xlabel('Number of Steps')
    plt.ylabel('CPU Time (s)')
    plt.title('CPU Time vs Number of Steps for Different Simulations')
    plt.legend()
    plt.show()
    
    # GPU Time vs Steps
    plt.figure()
    for simulation in df['simulation'].unique():
        subset = df[df['simulation'] == simulation]
        plt.scatter(subset['n_steps'], subset['gpu_time'], label=f'GPU - {simulation}', alpha=0.6)
    plt.xlabel('Number of Steps')
    plt.ylabel('GPU Time (s)')
    plt.title('GPU Time vs Number of Steps for Different Simulations')
    plt.legend()
    plt.show()
    
    # CPU Time vs Variants
    plt.figure()
    for simulation in df['simulation'].unique():
        subset = df[df['simulation'] == simulation]
        plt.scatter(subset['n_variants'], subset['cpu_time'], label=f'CPU - {simulation}', alpha=0.6)
    plt.xlabel('Number of Variants')
    plt.ylabel('CPU Time (s)')
    plt.title('CPU Time vs Number of Variants for Different Simulations')
    plt.legend()
    plt.show()
    
    # GPU Time vs Variants
    plt.figure()
    for simulation in df['simulation'].unique():
        subset = df[df['simulation'] == simulation]
        plt.scatter(subset['n_variants'], subset['gpu_time'], label=f'GPU - {simulation}', alpha=0.6)
    plt.xlabel('Number of Variants')
    plt.ylabel('GPU Time (s)')
    plt.title('GPU Time vs Number of Variants for Different Simulations')
    plt.legend()
    plt.show()

# Call the plotting function
plot_time_vs_steps_variants(df)
