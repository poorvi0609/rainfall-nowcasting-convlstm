import glob
import os

import numpy as np
import xarray as xr

LAT_RANGE = slice(16.0, 22.0)
LON_RANGE = slice(70.0, 76.0)


def load_cropped_frame(path):
    """Open one IMERG file and return its rainfall crop as a 2D array."""
    data = xr.open_dataset(path, group="Grid", engine="h5netcdf")
    rain = data["precipitation"].isel(time=0)
    rain = rain.sel(lat=LAT_RANGE, lon=LON_RANGE)
    rain = rain.transpose("lat", "lon")
    frame = rain.values
    timestamp = str(data["time"].values[0])
    data.close()
    return frame, timestamp


def main():
    # File names contain the date and time, so sorting puts them in order
    files = sorted(glob.glob("data/raw/*.HDF5"))

    frames = []
    timestamps = []
    for path in files:
        frame, timestamp = load_cropped_frame(path)
        frames.append(frame)
        timestamps.append(timestamp)

    rain = np.stack(frames)  # shape: (time, lat, lon)
    print("Array shape:", rain.shape)
    print("Missing values (NaN):", int(np.isnan(rain).sum()))
    print("Min / max / mean:", np.nanmin(rain), np.nanmax(rain), np.nanmean(rain))

    rainy_share = np.mean(rain > 0.1)
    print(f"Share of pixels with rain above 0.1 mm/hr: {rainy_share:.1%}")

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/rain_frames.npy", rain)
    np.save("data/processed/timestamps.npy", np.array(timestamps))


if __name__ == "__main__":
    main()