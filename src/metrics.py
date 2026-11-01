import numpy as np


def to_mm(x):
    """Undo the log1p transform to get rain rate in mm/hr."""
    return np.expm1(x)


def mse(pred, truth):
    return float(np.mean((pred - truth) ** 2))


def csi_pod(pred, truth, threshold):
    """CSI and POD for rain at or above a threshold (mm/hr)."""
    pred_rain = pred >= threshold
    true_rain = truth >= threshold

    hits = np.sum(pred_rain & true_rain)
    misses = np.sum(~pred_rain & true_rain)
    false_alarms = np.sum(pred_rain & ~true_rain)

    total_events = hits + misses + false_alarms
    csi = hits / total_events if total_events > 0 else float("nan")
    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    return csi, pod


def evaluate(pred_mm, truth_mm, name):
    """Print MSE, CSI and POD at each lead time. Inputs are in mm/hr."""
    print(f"--- {name} ---")
    for step, minutes in enumerate([30, 60]):
        pred = pred_mm[:, step, :, :, 0]
        truth = truth_mm[:, step, :, :, 0]
        line = f"{minutes} min ahead | MSE {mse(pred, truth):.3f}"
        for threshold in [1, 5]:
            csi, pod = csi_pod(pred, truth, threshold)
            line += f" | CSI@{threshold} {csi:.3f} POD@{threshold} {pod:.3f}"
        print(line)