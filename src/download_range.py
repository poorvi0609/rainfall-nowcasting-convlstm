import glob
import os
import shutil
from datetime import date, timedelta

import earthaccess
import numpy as np

from build_dataset import load_cropped_frame

BOUNDING_BOX = (70.0, 16.0, 76.0, 22.0)
START_DAY = date(2024, 7, 3)
END_DAY = date(2024, 7, 31)

TEMP_FOLDER = "data/raw_tmp"
OUTPUT_FOLDER = "data/processed/days"


def process_one_day(day):
    """Download one day, crop it, save it, and delete the raw files."""
    output_path = f"{OUTPUT_FOLDER}/{day}.npy"
    if os.path.exists(output_path):
        print(f"{day}: already done, skipping")
        return

    results = earthaccess.search_data(
        short_name="GPM_3IMERGHH",
        version="07",
        temporal=(f"{day} 00:00:00", f"{day} 23:59:59"),
        bounding_box=BOUNDING_BOX,
    )
    print(f"{day}: found {len(results)} files (expected 48)")
    if len(results) != 48:
        print(f"{day}: unexpected file count, skipping this day")
        return

    earthaccess.download(results, TEMP_FOLDER)

    files = sorted(glob.glob(f"{TEMP_FOLDER}/*.HDF5"))
    frames = []
    for path in files:
        frame, timestamp = load_cropped_frame(path)
        frames.append(frame)

    day_array = np.stack(frames)  # shape: (48, 60, 60)
    np.save(output_path, day_array)
    print(f"{day}: saved array with shape {day_array.shape}")

    shutil.rmtree(TEMP_FOLDER)


def main():
    earthaccess.login()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    day = START_DAY
    while day <= END_DAY:
        process_one_day(day)
        day += timedelta(days=1)


if __name__ == "__main__":
    main()