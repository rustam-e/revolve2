import psutil
import time

def track_cpu_usage():
    print("Tracking CPU usage for each core. Press Ctrl+C to stop.")
    while True:
        # Get the CPU usage percentage for each core
        per_core_usage = psutil.cpu_percent(interval=1, percpu=True)
        
        # Print the usage for each core
        print("Per-Core CPU Usage:")
        for i, usage in enumerate(per_core_usage):
            print(f"  Core {i}: {usage}%")
        
        # Optional: Separate each sample visually
        print("-" * 30)
        time.sleep(1)  # Adjust the interval if needed

if __name__ == "__main__":
    try:
        track_cpu_usage()
    except KeyboardInterrupt:
        print("\nCPU tracking stopped.")
