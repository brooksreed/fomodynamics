"""Functions for generating test data."""

import numpy as np


def generate_random_data(filename: str, num_streams: int, time_step: float, end_time: float):
    """Generate random time series data and save to CSV."""
    times = np.arange(0, end_time, time_step)
    data = np.random.randn(len(times), num_streams)

    header = ["time"] + [f"data{i+1}" for i in range(num_streams)]

    with open(filename, "w") as f:
        f.write(",".join(header) + "\n")
        for i, t in enumerate(times):
            row = [f"{t:.3f}"] + [f"{data[i, j]:.6f}" for j in range(num_streams)]
            f.write(",".join(row) + "\n")

    print(f"Saved {len(times)} rows to {filename}")
