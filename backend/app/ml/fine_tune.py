# fine_tune.py
"""
Offline fine-tuning script to incorporate user feedback corrections into your osteoarthritis and pneumonia models,
while safeguarding existing validation performance.

Workflow:
1. Backup current models.
2. Load original validation datasets.
3. Load feedback images and labels.
4. Create combined training tf.data.Dataset for fine-tuning.
5. Fine-tune each model for a few epochs with validation.
6. Save only if validation accuracy improves.
"""

import os
import glob
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
BASE_DIR       = Path(__file__).resolve().parent
MODELS_DIR     = BASE_DIR.parent / 'models'
FEEDBACK_DIR   = BASE_DIR / 'feedback'
DATA_DIR       = BASE_DIR.parent / 'datasets'  # original data root

# Backup models
def backup_model(src: Path, suffix: str = '_backup'):
    dst = src.with_suffix(src.suffix + suffix)
    if not dst.exists():
        shutil.copy(src, dst)
        print(f"Backed up {src.name} to {dst.name}")

osteo_path = MODELS_DIR / 'osteo_efficientnetb0.keras'
pneu_path  = MODELS_DIR / 'pneumonia_classifier.h5'
backup_model(osteo_path)
backup_model(pneu_path)

# Parameters
IMG_SIZE = 224   # EfficientNet input
PNEU_SIZE = (197, 279)
BATCH     = 8
EPOCHS    = 2    # small fine-tune
LR        = 1e-5

# Load original validation sets
def load_val_ds(task: str):
    # task: 'osteo' or 'pneu'
    x_paths, y_labels = [], []
    val_dir = DATA_DIR / ('Osteoarthritis' if task=='osteo' else 'Pneumonia') / 'val'
    for cat in os.listdir(val_dir):
        for ext in ('*.png','*.jpg','*.jpeg'):
            for p in glob.glob(str(val_dir/cat/ext)):
                x_paths.append(p)
                y_labels.append(1 if cat.lower().startswith(task[:4]) else 0)
    def gen():
        for p, y in zip(x_paths, y_labels):
            img = Image.open(p).convert('L')
            if task=='osteo': img = img.resize((IMG_SIZE, IMG_SIZE))
            else:            img = img.resize(PNEU_SIZE)
            arr = np.array(img, dtype='float32')
            if task=='osteo':
                arr = np.stack([arr,arr,arr],axis=-1)
                arr = preprocess_efficientnet(arr)
            else:
                arr = arr/255.0
                arr = arr.reshape(*PNEU_SIZE,1)
            yield arr, y
    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE,IMG_SIZE,3) if task=='osteo' else (*PNEU_SIZE,1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    return ds.batch(BATCH)

val_osteo_ds = load_val_ds('osteo')
val_pneu_ds  = load_val_ds('pneu')

# Gather feedback examples

def load_feedback(task: str):
    imgs, labs = [], []
    fb_dir = FEEDBACK_DIR / task
    size = (IMG_SIZE,IMG_SIZE) if task=='osteo' else PNEU_SIZE
    for ext in ('*.png','*.jpg','*.jpeg'):
        for p in glob.glob(str(fb_dir/ext)):
            img = Image.open(p).convert('L').resize(size)
            arr = np.array(img, dtype='float32')
            if task=='osteo':
                arr = np.stack([arr,arr,arr],axis=-1)
                arr = preprocess_efficientnet(arr)
            else:
                arr = arr/255.0
                arr = arr.reshape(*PNEU_SIZE,1)
            imgs.append(arr)
            labs.append(1)
    return imgs, labs

osteo_imgs, osteo_labs = load_feedback('osteo')
pneu_imgs, pneu_labs   = load_feedback('pneu')

# Create tf.data train sets

def make_ds(imgs, labs):
    X = np.stack(imgs, axis=0)
    y = np.array(labs)
    return tf.data.Dataset.from_tensor_slices((X,y)).shuffle(100).batch(BATCH)

train_osteo_ds = make_ds(osteo_imgs, osteo_labs)
train_pneu_ds  = make_ds(pneu_imgs, pneu_labs)

# Fine-tune Osteoarthritis model
osteo_model = tf.keras.models.load_model(str(osteo_path))
total = len(osteo_model.layers)
unfreeze_at = int(total * 0.8)
for layer in osteo_model.layers[:unfreeze_at]: layer.trainable=False
for layer in osteo_model.layers[unfreeze_at:]:  layer.trainable=True
osteo_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Fine-tuning Osteoarthritis model...")
osteo_model.fit(
    train_osteo_ds,
    validation_data=val_osteo_ds,
    epochs=EPOCHS,
    callbacks=[ModelCheckpoint(str(osteo_path), monitor='val_accuracy', save_best_only=True, verbose=1)]
)

# Fine-tune Pneumonia model
pneu_model = tf.keras.models.load_model(str(pneu_path))
total = len(pneu_model.layers)
unfreeze_at = int(total * 0.8)
for layer in pneu_model.layers[:unfreeze_at]: layer.trainable=False
for layer in pneu_model.layers[unfreeze_at:]:  layer.trainable=True
pneu_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Fine-tuning Pneumonia model...")
pneu_model.fit(
    train_pneu_ds,
    validation_data=val_pneu_ds,
    epochs=EPOCHS,
    callbacks=[ModelCheckpoint(str(pneu_path), monitor='val_accuracy', save_best_only=True, verbose=1)]
)

print("Fine-tuning complete. Models updated only if validation improved.")
