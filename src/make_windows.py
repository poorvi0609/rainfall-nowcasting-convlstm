import glob
import os
from datetime import date, timedelta

import numpy as np

INPUT_STEPS = 6    # 3 hours of history
OUTPUT_STEPS = 2   # predict the next 30 and 60 minutes
GAP = INPUT_STEPS + OUTPUT_STEPS  # windows skipped between splits


def load_segments():
    """Load all frames, keeping runs of consecutive days together."""
    segments = [np.load("data/processed/rain_frames.npy")]  # 1 and 2 July
    last_day = date(2024, 7, 2)

    for path in sorted(glob.glob("data/processed/days/*.npy")):
        day = date.fromisoformat(os.path.basename(path)[:-4])
        data = np.load(path)
        if day == last_day + timedelta(days=1):
            segments[-1] = np.concatenate([segments[-1], data])
        else:
            segments.append(data)  # a day is missing, so start a new run
        last_day = day
    return segments


def make_windows(frames):
    """Cut one run of frames into (input, target) pairs."""
    inputs, targets = [], []
    total = INPUT_STEPS + OUTPUT_STEPS
    for start in range(len(frames) - total + 1):
        inputs.append(frames[start:start + INPUT_STEPS])
        targets.append(frames[start + INPUT_STEPS:start + total])
    return inputs, targets


def main():
    all_inputs, all_targets = [], []
    for segment in load_segments():
        inputs, targets = make_windows(segment)
        all_inputs += inputs
        all_targets += targets

    x = np.array(all_inputs)
    y = np.array(all_targets)

    # Rain is very skewed, so compress it with log(1 + rain)
    x = np.log1p(x)[..., np.newaxis]   # shape (N, 6, 60, 60, 1)
    y = np.log1p(y)[..., np.newaxis]   # shape (N, 2, 60, 60, 1)

    # Split by time: first 70% train, next 15% validation, last 15% test
    n = len(x)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Skip GAP windows at each boundary so no frame appears on both sides
    splits = {
        "x_train": x[:train_end], "y_train": y[:train_end],
        "x_val": x[train_end + GAP:val_end], "y_val": y[train_end + GAP:val_end],
        "x_test": x[val_end + GAP:], "y_test": y[val_end + GAP:],
    }
    for name, array in splits.items():
        print(name, array.shape)

    np.savez_compressed("data/processed/windows.npz", **splits)


if __name__ == "__main__":
    main()