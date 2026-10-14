"""Split labelled frames into train/val sets in the layout YOLO expects.

    python training/prepare_dataset.py         # training/data/labelled -> training/data/dataset

Expects a YOLO-format export from your labelling tool in training/data/labelled/:
    images/<name>.jpg
    labels/<name>.txt   one line per mosquito: "0 x_center y_center width height" (0-1)
Images without a label file are kept as empty frames: they teach the model what
an empty chamber looks like, which cuts false positives.
"""
import os
import random
import shutil

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
LABELLED_DIR = os.path.join(TRAINING_DIR, "data", "labelled")
DATASET_DIR = os.path.join(TRAINING_DIR, "data", "dataset")
VAL_FRACTION = 0.2
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    images = sorted(f for f in os.listdir(os.path.join(LABELLED_DIR, "images")) if f.lower().endswith(IMAGE_EXTENSIONS))
    if not images:
        raise SystemExit(f"No images found in {LABELLED_DIR}/images")

    random.Random(0).shuffle(images)  # fixed seed: the same split every run
    val_count = max(1, round(len(images) * VAL_FRACTION))
    splits = {"val": images[:val_count], "train": images[val_count:]}

    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    unlabelled = 0
    for split, names in splits.items():
        os.makedirs(os.path.join(DATASET_DIR, "images", split))
        os.makedirs(os.path.join(DATASET_DIR, "labels", split))
        for name in names:
            label = os.path.splitext(name)[0] + ".txt"
            shutil.copy(os.path.join(LABELLED_DIR, "images", name), os.path.join(DATASET_DIR, "images", split, name))
            source_label = os.path.join(LABELLED_DIR, "labels", label)
            target_label = os.path.join(DATASET_DIR, "labels", split, label)
            if os.path.exists(source_label):
                shutil.copy(source_label, target_label)
            else:
                open(target_label, "w").close()
                unlabelled += 1

    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(f"path: {DATASET_DIR.replace(os.sep, '/')}\n")
        f.write("train: images/train\nval: images/val\nnames:\n  0: mosquito\n")

    print(f"{len(splits['train'])} train / {len(splits['val'])} val images "
          f"({unlabelled} without labels, kept as empty frames) -> {DATASET_DIR}")


if __name__ == "__main__":
    main()
