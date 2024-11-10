import argparse
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor 
import subprocess
import csv
import mujoco
import psutil
import math
# Import the MuJoCo XML models from the new filea
from mujoco_models import _XML_ANT, _XML_BALL, _XML_HUMANOID

_CPU_COUNT = multiprocessing.cpu_count()

# Set the start method to spawn to avoid fork-related issues with JAX
multiprocessing.set_start_method("spawn", force=True)

def _cpu_profile_inner_batched(model_xml: str, n_steps: int, batch_size: int):
    from mujoco import MjModel, MjData, mj_step
    model = MjModel.from_xml_string(model_xml)

    for _ in range(batch_size):
        data = MjData(model)
        data.ctrl = 1
        for _ in range(n_steps):
            mj_step(model, data)

def track_cpu_usage(profiling_event, cpu_usage_samples):
    """Function to track CPU usage and store it in cpu_usage_samples."""
    while profiling_event.is_set():
        per_core_usage = psutil.cpu_percent(interval=1, percpu=True)
        avg_usage = sum(per_core_usage) / len(per_core_usage)  # Average across all cores
        cpu_usage_samples.append(avg_usage)

def cpu_profile_batched(model_xml: str, n_variants: int, n_steps: int, max_processes: int, batch_size: int = 10):
    print(f"Running CPU profile with {n_variants=}, {n_steps=}, {max_processes=}, {batch_size=}")
    assert 0 < max_processes <= _CPU_COUNT

    # Track CPU usage across all cores
    cpu_usage_samples = multiprocessing.Manager().list()
    profiling_event = multiprocessing.Event()
    profiling_event.set()

    # Start tracking CPU usage in a separate process
    cpu_tracker = multiprocessing.Process(target=track_cpu_usage, args=(profiling_event, cpu_usage_samples))
    cpu_tracker.start()

    # Perform profiling
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_processes) as pool:
        tasks = [pool.submit(_cpu_profile_inner_batched, model_xml, n_steps, min(batch_size, n_variants - i))
                 for i in range(0, n_variants, batch_size)]
        _ = [task.result() for task in tasks]
    total_time = time.perf_counter() - start_time

    # Stop tracking CPU usage
    profiling_event.clear()
    cpu_tracker.join()

    # Calculate average CPU usage across all cores
    avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0

    print(f"Average CPU Usage during profiling (across all cores): {avg_cpu_usage:.2f}%")
    return total_time, avg_cpu_usage

def get_available_gpu_memory():
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                                         shell=True).decode().strip()
        return int(output.splitlines()[0]) 
    except Exception as e:
        print(f"Error checking available GPU memory: {e}")
        return None


def compare_combined(model_xml: str, n_variants: int, n_steps: int, max_processes: int, gpu_cpu_ratio: float, sim_name: str):
    # Calculate the split based on ratio
    gpu_variants = math.floor(n_variants * gpu_cpu_ratio)
    cpu_variants = n_variants - gpu_variants

    # Start the timer before launching tasks
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        cpu_future = pool.submit(cpu_profile_batched, model_xml, cpu_variants, n_steps, max_processes // 2)
        gpu_future = pool.submit(gpu_profile, model_xml, gpu_variants, n_steps)
        
        # Retrieve results when both tasks complete
        cpu_time, avg_cpu_usage = cpu_future.result()
        gpu_time, gpu_utilization = gpu_future.result()

    # End the timer after both tasks complete
    total_time = time.perf_counter() - start_time
    
    return {
        "simulation_name": sim_name,
        "gpu_cpu_ratio":gpu_cpu_ratio,
        "total_time": total_time,
        "combined_cpu_time": cpu_time,
        "combined_gpu_time": gpu_time,
        "combined_avg_cpu_usage": avg_cpu_usage,
        "combined_gpu_utilization": gpu_utilization,
        "n_gpu_variants": gpu_variants,
        "n_cpu_variants": cpu_variants
    }

def compare_sequential(model_xml: str, n_variants: int, n_steps: int, max_processes: int, sim_name: str):
    cpu_time, avg_cpu_usage = cpu_profile_batched(model_xml, n_variants, n_steps, max_processes)
    gpu_time, gpu_utilization = gpu_profile(model_xml, n_variants, n_steps)
    print('cpu_cpu_ratio: ', gpu_cpu_ratio)

    # Determine which is faster
    gpu_win = "better" if gpu_time < cpu_time else "worse"
    print('gpu_cpu_ratio: ', gpu_cpu_ratio)
    faster, slower = (cpu_time, gpu_time)[::2 * (gpu_time > cpu_time) - 1]
    percentage = int(100 * (slower / faster - 1))
    
    gpu_cpu_ratio = 1 - gpu_time / (gpu_time + cpu_time)
    print('gpu_cpu_ratio: ', gpu_cpu_ratio)
    
    return {
        "simulation_name": sim_name,
        "n_variants": n_variants,
        "n_steps": n_steps,
        "cpu_time": cpu_time,
        "gpu_time": gpu_time,
        "gpu_win": gpu_win,
        "speed_difference": percentage,
        "gpu_utilization": gpu_utilization,
        "avg_cpu_usage": avg_cpu_usage, 
        "gpu_cpu_ratio": gpu_cpu_ratio
    }

# def compare(model_xml: str, n_variants: int, n_steps: int, max_processes: int, sim_name: str):
#     # to do - re-enable once issue of failing gpu is fixed

    
#     # cpu_time= 10 # to do - remove
#     # gpu_time = 10 # to do - remove
    
#     gpu_cpu_ratio = 1 - gpu_time / (gpu_time + cpu_time)
#     total_time, combined_cpu_time, combined_gpu_time, combined_avg_cpu_usage, combined_gpu_utilization, gpu_variants, cpu_variants  = compare_combined(model_xml, n_variants, n_steps, max_processes, gpu_cpu_ratio)
      
#     # Determine which is faster
#     gpu_win = "better" if gpu_time < cpu_time else "worse"
#     faster, slower = (cpu_time, gpu_time)[::2 * (gpu_time > cpu_time) - 1]
#     percentage = int(100 * (slower / faster - 1))
    

#     return {
#         "simulation_name": sim_name,
#         "n_variants": n_variants,
#         "n_steps": n_steps,
#         "cpu_time": cpu_time,
#         "gpu_time": gpu_time,
#         "gpu_win": gpu_win,
#         "speed_difference": percentage,
#         # "gpu_utilization": gpu_utilization, - to do reenable
#         # "avg_cpu_usage": avg_cpu_usage, - to do reenable
#         "total_time": total_time,
#         "combined_cpu_time": combined_cpu_time,
#         "combined_gpu_time": combined_gpu_time,
#         "combined_avg_cpu_usage": combined_avg_cpu_usage,
#         "gpu_cpu_ratio": gpu_cpu_ratio,
#         "combined_gpu_utilization": combined_gpu_utilization,
#         "n_gpu_variants": gpu_variants,
#         "n_cpu_variants": cpu_variants
        
#     }

def write_to_csv(filename, data, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode, newline="") as file:
        writer = csv.writer(file)

        # If not appending, write the header for the sequential results
        if not append:
            writer.writerow([
                "simulation_name",
                 "n_variants",
                 "n_steps",
                 "cpu_time",
                 "gpu_time",
                "gpu_win",
                 "speed_difference",
                 "gpu_utilization",
                 "avg_cpu_usage",
                 "gpu_cpu_ratio"
            ])
            for entry in data:
                writer.writerow([
                    entry["simulation_name"],
                    entry["n_variants"],
                    entry["n_steps"],
                    entry["cpu_time"],
                    entry["gpu_time"],
                    entry["gpu_win"],
                    entry["speed_difference"],
                    entry["gpu_utilization"],
                    entry["avg_cpu_usage"],
                    entry["gpu_cpu_ratio"]
                ])
        else:
            # Appending combined results with a different header
            writer.writerow([
                "simulation_name",
                 "total_time",
                 "combined_cpu_time",
                 "combined_gpu_time",
                "combined_avg_cpu_usage",
                 "combined_gpu_utilization",
                 "gpu_cpu_ratio",
                "n_gpu_variants",
                 "n_cpu_variants"
            ])
            for entry in data:
                writer.writerow([
                    entry["simulation_name"],
                    entry["total_time"],
                    entry["combined_cpu_time"],
                    entry["combined_gpu_time"],
                    entry["combined_avg_cpu_usage"],
                    entry["combined_gpu_utilization"],
                    entry["gpu_cpu_ratio"],
                    entry["n_gpu_variants"],
                    entry["n_cpu_variants"]
                ])
                
def gpu_profile(model_xml: str, n_variants: int, n_steps: int):
    print(f"Running GPU profile with {n_variants=}, {n_steps=} ...")
    from mujoco import mjx
    import jax
    import jax.numpy as jnp
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    mjx_datas = jax.vmap(lambda _: mjx_data.replace(ctrl=1))(jnp.arange(n_variants))
    step = jax.vmap(jax.jit(mjx.step), in_axes=(None, 0))

    # do not even time first step, takes too long, as long as step 2...N is faster we have viable option for GPU
    # because first step is fixed cost
    mjx_datas = step(mjx_model, mjx_datas)

    t = time.perf_counter()

    for i_steps in range(n_steps - 1):
        mjx_datas = step(mjx_model, mjx_datas)
    elapsed = time.perf_counter() - t

    # Capture peak GPU utilization using nvidia-smi
    peak_gpu_utilization = get_peak_gpu_utilization()
    
    return elapsed, peak_gpu_utilization

def get_peak_gpu_utilization():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            shell=True
        ).decode().strip()
        peak_utilization = max(map(int, output.splitlines()))
        return peak_utilization
    except Exception as e:
        print(f"Error capturing GPU utilization: {e}")
        return 0  # Default to 0 if utilization cannot be retrieved


def log_result(result):
    log_message = (
        f"Simulation '{result['simulation_name']}' completed:\n"
        # f"  Variants: {result['n_variants']}, Steps: {result['n_steps']}\n"
        # f"  CPU Time: {result['cpu_time']:.4f} s, GPU Time: {result['gpu_time']:.4f} s\n"
        # f"  GPU is {result['gpu_win']} than CPU by {result['speed_difference']}%\n"
        # f"  GPU Utilization: {result['gpu_utilization']}%\n"
        # f"  CPU Utilization: {result['avg_cpu_usage']}%\n"
        f"  total_time: {result['total_time']}\n"
        f"  combined_cpu_time: {result['combined_cpu_time']}\n"
        f"  combined_gpu_time: {result['combined_gpu_time']}\n"
        f"  combined_avg_cpu_usage: {result['combined_avg_cpu_usage']}%\n"
        f"  combined_gpu_utilization: {result['combined_gpu_utilization']}%\n"
        f"  gpu_cpu_ratio: {result['gpu_cpu_ratio']* 100}%\n"
        f"  n_gpu_variants: {result['n_gpu_variants']}\n"
        f"  n_cpu_variants: {result['n_cpu_variants']}\n"
    )
    print(log_message)
    
def main(simulations, max_processes=None):
    if max_processes is None:
        max_processes = multiprocessing.cpu_count()
    
    # variants = [32, 1024, 2056, 4096, 8192, 16384, 32768, 65536, 131072, 256000, 512000, 1000000,2000000, 4000000]
    # steps = [32, 100, 500, 1024, 2000, 4000]
    variants = [32, 1024]
    steps = [100]
    results = []
    
    parser = argparse.ArgumentParser(description="Benchmark CPU and GPU profiling for MuJoCo models.")
    parser.add_argument("benchmark_type", choices=["sequential", "combined"], help="Type of benchmark to run")
    parser.add_argument("--file", default="performance_metrics.csv", help="CSV file to save results")
    args = parser.parse_args()
    
    if args.benchmark_type == "sequential":

        # Loop through each simulation
        for sim_name, model_xml in simulations.items():
            print(f"Running benchmarks for simulation: {sim_name}")
            for n_variants in variants:
                for n_steps in steps:
                    try:
                        # Sequential profiling
                        sequential_results = compare_sequential(model_xml, n_variants, n_steps, max_processes, sim_name)
                        gpu_cpu_ratio = sequential_results["gpu_cpu_ratio"]
      
                        # Log sequential results
                        results.append(sequential_results)
                        write_to_csv("performance_metrics.csv", results, append=False)
                        log_result(results) 
                    except Exception as e:
                        print(f"Error with {sim_name}, n_variants={n_variants}, n_steps={n_steps}: {e}")
    elif args.benchmark_type == "combined":
        with open(args.file, mode="r") as file:
            reader = csv.DictReader(file)
            ratios = {
                (row["simulation_name"], int(row["n_variants"]), int(row["n_steps"])): float(row["gpu_cpu_ratio"])
                for row in reader
            }
        for sim_name, model_xml in simulations.items():
            for n_variants in variants:
                for n_steps in steps:
                    key = (sim_name, n_variants, n_steps)
                    if key in ratios:
                        gpu_cpu_ratio = ratios[key]
                        # Combined profiling using the ratio
                        combined_results = compare_combined(model_xml, n_variants, n_steps, max_processes, gpu_cpu_ratio, sim_name)
                        
                        # Log combined results
                        results.append(combined_results)
                        write_to_csv("performance_metrics.csv", results, append=True)
                        log_result(combined_results) 
            
    # Write all results to CSV
    print("All results written to performance_metrics.csv")


if __name__ == '__main__':
    # Define simulations to benchmark
    simulations = {
        # "ant": _XML_ANT,
        "ball": _XML_BALL,
        # "humanoid": _XML_HUMANOID,
    }
    main(simulations)