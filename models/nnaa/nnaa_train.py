import os
from os.path import isfile, isdir, join
from multiprocessing import freeze_support
import random
import sys
import tensorflow as tf
import numpy as np
from PIL import Image

# Enable mixed precision (float16 compute, float32 accumulation)
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def extract_luma(img_path):
    img = Image.open(img_path)
    noAA_image = img.split()

    r = np.float32(noAA_image[0])
    g = np.float32(noAA_image[1])
    b = np.float32(noAA_image[2])
    y = r * 0.299 + g * 0.587 + b * 0.114

    noAA_tensor_luma = y.reshape(noAA_image[0].size[1], noAA_image[0].size[0], 1) / 255

    img.close()
    return noAA_tensor_luma


class NnaaDataset(tf.keras.utils.PyDataset):
    def __init__(self, bases_dir, targets_dir, batch_size, use_cache=False, **kwargs):
        super().__init__(**kwargs)
        self.img_names = list(set(os.listdir(bases_dir)).union(os.listdir(targets_dir)))
        self.img_names = [f for f in self.img_names if isfile(join(bases_dir, f)) and isfile(join(targets_dir, f))]

        random.shuffle(self.img_names)

        self.bases_dir = bases_dir
        self.targets_dir = targets_dir
        self.batch_size = batch_size

        self.cache_built = False
        if use_cache:
            self.cache_tensors = []
            for i in range(len(self)):
                self.cache_tensors.append(self[i])
            self.cache_built = True

    def __len__(self):
        return len(self.img_names) // self.batch_size

    def __getitem__(self, idx):
        if self.cache_built:
            return self.cache_tensors[idx]

        inputs = []
        targets = []
        r_idx = idx * self.batch_size

        for i in range(r_idx, r_idx + self.batch_size):
            img_name = self.img_names[i]
            base_path = join(self.bases_dir, img_name)
            target_path = join(self.targets_dir, img_name)

            x = extract_luma(base_path)
            y = extract_luma(target_path)

            inputs.append(x)
            targets.append(y - x)

        # Return as float16 (mixed precision will cast to float16 internally)
        return (np.half(inputs), np.half(targets))


def residual_block(x, filters, kernel_size=3, strides=1):
    """Residual block with two conv layers and skip connection."""
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # If dimensions differ, adjust shortcut with 1x1 conv
    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


# Custom callback to print batch losses in real time
class BatchLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_batches):
        super().__init__()
        self.total_batches = total_batches
        self.batch_idx = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_idx += 1
        loss = logs.get('loss')
        # Print loss for every batch, flush to output immediately
        print(f"  Batch {self.batch_idx}/{self.total_batches}: loss = {loss:.8f}")
        sys.stdout.flush()


if __name__ == "__main__":
    freeze_support()

    base_dir_path = "data/train/bad/1280x720"
    target_dir_path = "data/train/fixed/1280x720"
    base_dir_path_test = "data/test/bad/2560x1440"
    target_dir_path_test = "data/test/fixed/2560x1440"

    model_name = "nnaa"
    models_path = ".."
    lr = 0.0001
    batch_size_train = 8
    batch_size_val = 4

    # ------------------------------
    # Improved architecture (mixed precision compatible)
    # ------------------------------
    input_img = tf.keras.Input(shape=(None, None, 1), name="img")

    # ---- Detail branch (two 3x3 convs, deeper) ----
    d = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(input_img)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.PReLU(shared_axes=[1, 2])(d)
    d = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(d)
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.PReLU(shared_axes=[1, 2])(d)

    # ---- Context branch with residual blocks ----
    c = tf.keras.layers.Conv2D(32, 8, strides=2, padding='same')(input_img)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.PReLU(shared_axes=[1, 2])(c)

    for _ in range(3):
        c = residual_block(c, 32, kernel_size=3)

    c = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(c)
    c = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(c)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.PReLU(shared_axes=[1, 2])(c)

    # ---- Fusion: concat + 3x3 + 1x1 ----
    concat = tf.keras.layers.Concatenate()([d, c])
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(concat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same')(x)

    # Cast output to float32 for loss computation
    output = tf.keras.layers.Activation('linear', dtype='float32')(x)

    # ------------------------------
    # Model creation
    # ------------------------------
    model_directory = join(models_path, model_name)
    if not isdir(model_directory):
        os.mkdir(model_directory)

    model_path = join(model_directory, model_name) + ".keras"
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    if isfile(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating new model...")
        model = tf.keras.Model(input_img, output, name=model_name)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            alpha=1e-6
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=['mean_absolute_error'])

    model.summary()

    if isfile(model_path) and model.optimizer.learning_rate != lr:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            alpha=1e-6
        )
        model.optimizer.learning_rate = lr_schedule
        print("Learning rate schedule updated")

    train_dataset = NnaaDataset(base_dir_path, target_dir_path, batch_size_train, use_cache=True)
    test_dataset = NnaaDataset(base_dir_path_test, target_dir_path_test, batch_size_val, use_cache=True)

    best_error_value = float('inf')
    if isfile(join(model_directory, "bestError.npy")):
        best_error_value = np.load(join(model_directory, "bestError.npy")).item()
    print(f"Best Error Value: {best_error_value}")

    patience = 10
    no_improve_epochs = 0

    # Training loop with real‑time batch progress
    epoch = 0
    while True:
        epoch += 1
        print(f"\nEpoch {epoch}")

        # Get number of batches for progress display
        total_batches = len(train_dataset)
        batch_callback = BatchLossCallback(total_batches)

        # Train one epoch with verbose=0 (we handle progress via callback)
        history = model.fit(train_dataset, epochs=1, verbose=0, callbacks=[batch_callback])

        # Evaluate on validation set
        eval_metrics = model.evaluate(test_dataset, verbose=2)
        current_loss = eval_metrics[0]

        print(f"Validation loss: {current_loss:.6f} (best: {best_error_value:.6f})")

        if current_loss < best_error_value:
            best_error_value = current_loss
            np.save(join(model_directory, "bestError"), best_error_value)
            model.save(model_path)
            print("Model saved (new best)")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        if epoch >= 100:
            print("Reached maximum epochs.")
            break
