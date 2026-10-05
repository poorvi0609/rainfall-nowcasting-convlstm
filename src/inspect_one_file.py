import glob

import matplotlib.pyplot as plt
import xarray as xr

# Same region we searched for: west coast of India
LAT_RANGE = slice(16.0, 22.0)
LON_RANGE = slice(70.0, 76.0)


def main():
    files = sorted(glob.glob("data/raw/*.HDF5"))
    print(f"{len(files)} files found")

    # Pick one file from the middle of the two days
    path = files[40]
    print("Opening:", path)

    # IMERG stores its data inside a group called "Grid"
    data = xr.open_dataset(path, group="Grid", engine="h5netcdf")
    print(data)  # shows the variables and dimensions available

    # Take the rainfall variable and crop it to our region
    rain = data["precipitation"].isel(time=0)
    rain = rain.sel(lat=LAT_RANGE, lon=LON_RANGE)
    rain = rain.transpose("lat", "lon")
    print("Cropped shape (lat, lon):", rain.shape)

    rain.plot(cmap="Blues", cbar_kwargs={"label": "Rain rate (mm/hr)"})
    plt.title("IMERG half-hourly rainfall, cropped region")
    plt.savefig("reports/sample_frame.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()