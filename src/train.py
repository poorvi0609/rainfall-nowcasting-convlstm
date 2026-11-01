import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from metrics import csi_pod, evaluate, mse, to_mm
from models import build_cnn, build_convlstm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "convlstm"], default="convlstm")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--data", default="data/processed/windows.npz")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


def append_results(path, run_name, pred_mm, truth_mm):
    """Add one row per lead time to a CSV that becomes the results table."""
    new_file = not os.path.exists(path)
    with open(path, "a") as f:
        if new_file:
            f.write("run,minutes,mse,csi1,pod1,csi5,pod5\n")
        for step, minutes in enumerate([30, 60]):
            pred = pred_mm[:, step, :, :, 0]
            truth = truth_mm[:, step, :, :, 0]
            csi1, pod1 = csi_pod(pred, truth, 1)
            csi5, pod5 = csi_pod(pred, truth, 5)
            f.write(f"{run_name},{minutes},{mse(pred, truth):.4f},"
                    f"{csi1:.4f},{pod1:.4f},{csi5:.4f},{pod5:.4f}\n")


def main():
    args = parse_args()
    run_name = (f"{args.model}_{args.optimizer}_lr{args.lr}"
                f"_do{args.dropout}_l2{args.l2}")
    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.data)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    if args.model == "cnn":
        model = build_cnn(dropout=args.dropout, l2=args.l2)
    else:
        model = build_convlstm(dropout=args.dropout, l2=args.l2)

    if args.optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)

    model.compile(optimizer=optimizer, loss="mse")

    # Stop when validation loss stops improving, and keep the best weights
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=8,
        callbacks=[early_stop],
    )

    # Loss curves for the report
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(run_name)
    plt.legend()
    plt.savefig(f"{args.out}/{run_name}_loss.png", dpi=150)

    # Score on the held-out test period, in mm/hr
    pred = to_mm(model.predict(x_test))
    truth = to_mm(y_test)
    evaluate(pred, truth, run_name)
    append_results(f"{args.out}/results.csv", run_name, pred, truth)

    with open(f"{args.out}/{run_name}_history.json", "w") as f:
        json.dump(history.history, f)


if __name__ == "__main__":
    main()