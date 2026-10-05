import earthaccess

# Region: west coast of India, as (west, south, east, north)
BOUNDING_BOX = (70.0, 16.0, 76.0, 22.0)
START_DATE = "2024-07-01"
END_DATE = "2024-07-02"


def main():
    # Asks for your Earthdata username and password the first time
    earthaccess.login()

    results = earthaccess.search_data(
        short_name="GPM_3IMERGHH",   # half-hourly IMERG rainfall
        version="07",
        temporal=(START_DATE, END_DATE),
        bounding_box=BOUNDING_BOX,
    )
    print(f"Found {len(results)} files")

    earthaccess.download(results, "data/raw")


if __name__ == "__main__":
    main()