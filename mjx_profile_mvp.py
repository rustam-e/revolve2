import argparse
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import csv
import math
import psutil
import mujoco

# Import your MuJoCo XML models
from mujoco_models import XML_BOX, XML_BOX_AND_BALL, XML_ARM_WITH_ROPE, XML_HUMANOID, XML_QUADROPED_VB

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
    import psutil
    while profiling_event.is_set():
        per_core_usage = psutil.cpu_percent(interval=1, percpu=True)
        avg_usage = sum(per_core_usage) / len(per_core_usage)
        cpu_usage_samples.append(avg_usage)

def cpu_profile_batched(model_xml: str, n_variants: int, n_steps: int,
                        max_processes: int, batch_size: int = 10):
    print(f"Running CPU profile with {n_variants=}, {n_steps=}, {max_processes=}, {batch_size=}")
    assert 0 < max_processes <= _CPU_COUNT

    # Track CPU usage across all cores
    cpu_usage_samples = multiprocessing.Manager().list()
    profiling_event = multiprocessing.Event()
    profiling_event.set()

    cpu_tracker = multiprocessing.Process(target=track_cpu_usage,
                                          args=(profiling_event, cpu_usage_samples))
    cpu_tracker.start()

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_processes) as pool:
        tasks = [
            pool.submit(_cpu_profile_inner_batched, model_xml, n_steps, min(batch_size, n_variants - i))
            for i in range(0, n_variants, batch_size)
        ]
        _ = [task.result() for task in tasks]
    total_time = time.perf_counter() - start_time

    profiling_event.clear()
    cpu_tracker.join()

    avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples) if cpu_usage_samples else 0

    print(f"Average CPU Usage during profiling (across all cores): {avg_cpu_usage:.2f}%")
    return total_time, avg_cpu_usage

def get_available_gpu_memory():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
            shell=True
        ).decode().strip()
        return int(output.splitlines()[0])
    except Exception as e:
        print(f"Error checking available GPU memory: {e}")
        return None

def gpu_profile(model_xml: str, n_variants: int, n_steps: int):
    print(f"Running GPU profile with {n_variants=}, {n_steps=} ...")
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    import mujoco

    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    mjx_datas = jax.vmap(lambda _: mjx_data.replace(ctrl=1))(jnp.arange(n_variants))
    step = jax.vmap(jax.jit(mjx.step), in_axes=(None, 0))

    # "Burn-in" first step
    mjx_datas = step(mjx_model, mjx_datas)

    # Now measure time for the remaining steps
    t = time.perf_counter()
    for _ in range(n_steps - 1):
        mjx_datas = step(mjx_model, mjx_datas)
    elapsed = time.perf_counter() - t

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
        return 0  # If utilization cannot be retrieved

def compare_sequential(model_xml: str, n_variants: int, n_steps: int,
                       max_processes: int, sim_name: str):
    cpu_time, avg_cpu_usage = cpu_profile_batched(model_xml, n_variants, n_steps, max_processes)
    gpu_time, gpu_utilization = gpu_profile(model_xml, n_variants, n_steps)

    gpu_win = "better" if gpu_time < cpu_time else "worse"
    faster, slower = (cpu_time, gpu_time)[::2 * (gpu_time > cpu_time) - 1]
    percentage = int(100 * (slower / faster - 1))

    gpu_cpu_ratio = 1 - gpu_time / (gpu_time + cpu_time)

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

def compare_combined(model_xml: str, n_variants: int, n_steps: int,
                     max_processes: int, gpu_cpu_ratio: float, sim_name: str):
    gpu_variants = math.floor(n_variants * gpu_cpu_ratio)
    cpu_variants = n_variants - gpu_variants

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        cpu_future = pool.submit(cpu_profile_batched, model_xml, cpu_variants, n_steps, max_processes // 2)
        gpu_future = pool.submit(gpu_profile, model_xml, gpu_variants, n_steps)

        cpu_time, avg_cpu_usage = cpu_future.result()
        gpu_time, gpu_utilization = gpu_future.result()

    total_time = time.perf_counter() - start_time

    return {
        "simulation_name": sim_name,
        "gpu_cpu_ratio": gpu_cpu_ratio,
        "total_time": total_time,
        "combined_cpu_time": cpu_time,
        "combined_gpu_time": gpu_time,
        "combined_avg_cpu_usage": avg_cpu_usage,
        "combined_gpu_utilization": gpu_utilization,
        "n_gpu_variants": gpu_variants,
        "n_cpu_variants": cpu_variants
    }

def write_sequential_csv(filename, data):
    """
    Writes sequential results to the specified CSV.
    """
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
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
    print(f"[Sequential] Wrote results to {filename}")

def write_combined_csv(filename, data):
    """
    Writes combined results to a different CSV.
    """
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
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
    print(f"[Combined] Wrote results to {filename}")

def log_sequential_result(result):
    log_message = (
        f"Simulation '{result['simulation_name']}' completed:\n"
        f"  Variants: {result['n_variants']}, Steps: {result['n_steps']}\n"
        f"  CPU Time: {result['cpu_time']:.4f} s, GPU Time: {result['gpu_time']:.4f} s\n"
        f"  GPU is {result['gpu_win']} than CPU by {result['speed_difference']}%\n"
        f"  GPU Utilization: {result['gpu_utilization']}%\n"
        f"  CPU Utilization: {result['avg_cpu_usage']}%\n"
    )
    print(log_message)

def log_combined_result(result):
    log_message = (
        f"Simulation '{result['simulation_name']}' completed:\n"
        f"  total_time: {result['total_time']}\n"
        f"  combined_cpu_time: {result['combined_cpu_time']}\n"
        f"  combined_gpu_time: {result['combined_gpu_time']}\n"
        f"  combined_avg_cpu_usage: {result['combined_avg_cpu_usage']}%\n"
        f"  combined_gpu_utilization: {result['combined_gpu_utilization']}%\n"
        f"  gpu_cpu_ratio: {result['gpu_cpu_ratio'] * 100:.2f}%\n"
        f"  n_gpu_variants: {result['n_gpu_variants']}\n"
        f"  n_cpu_variants: {result['n_cpu_variants']}\n"
    )
    print(log_message)

def run_sequential(simulations, simulation_variants, steps, repeat, seq_out_file, max_processes=None):
    """
    Runs sequential benchmarks and writes results to seq_out_file.
    """
    results = []
    for sim_name, model_xml in simulations.items():
        variants_list = simulation_variants.get(sim_name, [])
        print(f"\n[Sequential] Running benchmarks for simulation: {sim_name}")
        for n_variants in variants_list:
            for n_steps in steps:
                for run_i in range(repeat):
                    print(f"  -> Repeat {run_i+1}/{repeat} for {sim_name}, variants={n_variants}, steps={n_steps}")
                    try:
                        seq_res = compare_sequential(model_xml, n_variants, n_steps, max_processes, sim_name)
                        results.append(seq_res)
                        log_sequential_result(seq_res)
                    except Exception as e:
                        print(f"Error with {sim_name}, variants={n_variants}, steps={n_steps}: {e}")

    # Write all sequential results
    write_sequential_csv(seq_out_file, results)

def run_combined(simulations, simulation_variants, steps, repeat,
                 seq_file, comb_out_file, max_processes=None):
    """
    Runs combined benchmarks:
    - Reads sequential results (gpu_cpu_ratio) from seq_file
    - Writes combined results to comb_out_file
    """
    # 1. Read ratio from sequential CSV
    ratio_map = {}
    with open(seq_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row["simulation_name"]:
                continue
            key = (row["simulation_name"], int(row["n_variants"]), int(row["n_steps"]))
            ratio_val = float(row["gpu_cpu_ratio"])
            ratio_map[key] = ratio_val

    # 2. Run combined
    results_combined = []
    for sim_name, model_xml in simulations.items():
        variants_list = simulation_variants.get(sim_name, [])
        print(f"\n[Combined] Running benchmarks for simulation: {sim_name}")
        for n_variants in variants_list:
            for n_steps in steps:
                for run_i in range(repeat):
                    key = (sim_name, n_variants, n_steps)
                    gpu_cpu_ratio = ratio_map.get(key, None)
                    if gpu_cpu_ratio is None:
                        print(f"Skipping {sim_name}, {n_variants}, {n_steps}: no ratio found in {seq_file}.")
                        continue
                    print(f"  -> Repeat {run_i+1}/{repeat} for {sim_name}, variants={n_variants}, steps={n_steps}, ratio={gpu_cpu_ratio:.2f}")
                    try:
                        comb_res = compare_combined(model_xml, n_variants, n_steps,
                                                    max_processes, gpu_cpu_ratio, sim_name)
                        results_combined.append(comb_res)
                        log_combined_result(comb_res)
                    except Exception as e:
                        print(f"Error with combined run for {sim_name}, {n_variants}, {n_steps}: {e}")

    # 3. Write combined results
    write_combined_csv(comb_out_file, results_combined)

def main(simulations, simulation_variants, max_processes=None):
    if max_processes is None:
        max_processes = multiprocessing.cpu_count()

    steps = [1000]  # Example: single steps setting
    parser = argparse.ArgumentParser(description="Benchmark CPU & GPU with sequential or combined approach.")
    parser.add_argument("benchmark_type", choices=["sequential", "combined"], help="Type of benchmark to run")
    parser.add_argument("--repeat", type=int, default=3, help="How many times to repeat each config")
    
    # We'll hard-code the file paths for clarity: 
    # - performance_metrics.csv for sequential results
    # - performance_metrics_combined.csv for combined results
    # If you want to make them user-configurable, you could add more args here.
    
    args = parser.parse_args()

    if args.benchmark_type == "sequential":
        run_sequential(
            simulations=simulations,
            simulation_variants=simulation_variants,
            steps=steps,
            repeat=args.repeat,
            seq_out_file="performance_metrics.csv",
            max_processes=max_processes
        )
    else:  # combined
        run_combined(
            simulations=simulations,
            simulation_variants=simulation_variants,
            steps=steps,
            repeat=args.repeat,
            seq_file="performance_metrics.csv",           # read ratio from sequential results
            comb_out_file="performance_metrics_combined.csv",
            max_processes=max_processes
        )

    print("\nBenchmarking finished.")

if __name__ == '__main__':
    simulations = {
        "BOX": XML_BOX,
        "BOX_AND_BALL": XML_BOX_AND_BALL,
        "ARM_WITH_ROPE": XML_ARM_WITH_ROPE,
        "HUMANOID": XML_HUMANOID,
    }

    simulation_variants = {
        "BOX": [32, 128, 256, 512, 1024, 2056, 4096, 8192, 16384, 32768, 65536, 131072, 256000, 512000],
        "BOX_AND_BALL": [32, 128, 256, 512, 1024, 2056, 4096, 8192, 16384, 32768, 65536, 131072, 256000, 512000],
        "ARM_WITH_ROPE": [32, 128, 256, 512, 1024, 2056, 4096, 8192, 16384, 32768],
        "HUMANOID": [32, 128, 256, 512, 1024, 2056, 4096, 8192, 16384, 32768],
    }

    main(simulations, simulation_variants)
