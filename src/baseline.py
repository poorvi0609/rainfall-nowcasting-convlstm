import numpy as np

from metrics import evaluate, to_mm


def main():
    data = np.load("data/processed/windows.npz")
    x_test = data["x_test"]
    y_test = data["y_test"]

    # Persistence: repeat the last observed frame for both lead times
    last_frame = x_test[:, -1:, :, :, :]            # (N, 1, 60, 60, 1)
    persistence = np.repeat(last_frame, 2, axis=1)  # (N, 2, 60, 60, 1)

    evaluate(to_mm(persistence), to_mm(y_test), "Persistence baseline")


if __name__ == "__main__":
    main()