to run benchamrk:

python3.10 mjx_profile_mvp.py sequential --repeat 2
benchamrks first cpu and gpu for each combination of variants and steps and stores to performance_metrics.csv file

python3.10 mjx_profile_mvp.py combined --repeat 2

uses data from performance_metrics.csv file and re-runs with cpu and gpu in parallel  and  stores results in 
performance_metrics_combined.csv

analysis scripts
in 
seq_vs_combined_viz
sequential_viz_scripts
combined_viz_scripts

you can find respective scripts for generating charts analysing the performance data
currerntly requires performance data to be in root folder

for matplotlib scripts you need to pass arguments as follows: 

python3.10 combined_viz_scripts/read_stats-matploylib_combined.py --compare_variants