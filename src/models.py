from tensorflow import keras
from tensorflow.keras import layers, regularizers

INPUT_SHAPE = (6, 60, 60, 1)   # 6 frames of 60x60, 1 channel
OUTPUT_STEPS = 2               # next 30 and 60 minutes


def build_cnn(filters=32, dropout=0.0, l2=0.0):
    """CNN baseline: stack the 6 frames as channels and use 2D convolutions."""
    reg = regularizers.l2(l2) if l2 > 0 else None

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Permute((2, 3, 1, 4))(inputs)          # (60, 60, 6, 1)
    x = layers.Reshape((60, 60, 6))(x)                # frames become channels
    x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                      kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                      kernel_regularizer=reg)(x)
    x = layers.Conv2D(OUTPUT_STEPS, 1, activation="relu")(x)   # (60, 60, 2)
    x = layers.Permute((3, 1, 2))(x)                  # (2, 60, 60)
    outputs = layers.Reshape((OUTPUT_STEPS, 60, 60, 1))(x)
    return keras.Model(inputs, outputs, name="cnn")


def build_convlstm(filters=32, dropout=0.0, l2=0.0):
    """ConvLSTM: reads the 6 frames in order and keeps a memory of the motion."""
    reg = regularizers.l2(l2) if l2 > 0 else None

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.ConvLSTM2D(filters, 3, padding="same", return_sequences=True,
                          dropout=dropout, kernel_regularizer=reg)(inputs)
    x = layers.ConvLSTM2D(filters, 3, padding="same", return_sequences=False,
                          dropout=dropout, kernel_regularizer=reg)(x)  # (60, 60, filters)
    x = layers.Conv2D(OUTPUT_STEPS, 1, activation="relu")(x)           # (60, 60, 2)
    x = layers.Permute((3, 1, 2))(x)                                   # (2, 60, 60)
    outputs = layers.Reshape((OUTPUT_STEPS, 60, 60, 1))(x)
    return keras.Model(inputs, outputs, name="convlstm")