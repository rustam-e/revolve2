to run benchamrk:

python3.10 mjx_profile_mvp.py sequential --repeat 2
benchamrks first cpu and gpu for each combination of variants and steps and stores to performance_metrics.csv file

python3.10 mjx_profile_mvp.py combined --repeat 2

uses data from performance_metrics.csv file and re-runs with cpu and gpu in parallel and stores results in
performance_metrics_combined.csv

analysis scripts
in
seq_vs_combined_viz
sequential_viz_scripts
combined_viz_scripts

you can find respective scripts for generating charts analysing the performance data
currerntly requires performance data to be in root folder

for matplotlib scripts you need to pass arguments as follows: --compare_variants

for measuring sequential gpu and cpu runs:
python3.10 sequential_viz_scripts/read_stats-matploylib.py --compare_steps
python3.10 sequential_viz_scripts/read_stats-matploylib.py --compare_variants
python3.10 sequential_viz_scripts/analyze_gpu_data.py

for measuring combined gpu and cpu runs:

python3.10 combined_viz_scripts/read_stats-matploylib_combined.py --compare_variants
python3.10 combined_viz_scripts/analyze_gpu_data_combined_time_vs_variants_vs_gpu_util.py

for measuring sequential vs combined scripts:

python3.10 seq_vs_combined_viz/read_stats-matplotlib_seq_vs_combined.py
python3.10 seq_vs_combined_viz/analyze_total_time_vs_variants_vs_gpu_percentage.py
python3.10 seq_vs_combined_viz/analyze_gpu_data_combined.py
