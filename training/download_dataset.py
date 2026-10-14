"""Download the frames collect mode uploaded to Cloudinary, for labelling.

    python training/download_dataset.py        # -> training/data/raw/

Uses the Cloudinary keys in .env. Files already downloaded are skipped, so it can be
re-run as more frames arrive.
"""
import os

import cloudinary
import cloudinary.api
import requests
from dotenv import load_dotenv

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAINING_DIR)
DATASET_FOLDER = "mosqai_dataset"
RAW_DIR = os.path.join(TRAINING_DIR, "data", "raw")


def paginate(list_call, *args, **options):
    while True:
        page = list_call(*args, max_results=500, **options)
        yield from page["resources"]
        options["next_cursor"] = page.get("next_cursor")
        if not options["next_cursor"]:
            return


def list_dataset_frames():
    frames = list(paginate(cloudinary.api.resources, type="upload", prefix=f"{DATASET_FOLDER}/"))
    if frames:
        return frames
    # Accounts using Cloudinary's dynamic folders don't put the folder in the public_id.
    for subfolder in cloudinary.api.subfolders(DATASET_FOLDER)["folders"]:
        frames += paginate(cloudinary.api.resources_by_asset_folder, subfolder["path"])
    return frames


def main():
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )
    os.makedirs(RAW_DIR, exist_ok=True)

    frames = list_dataset_frames()
    downloaded = 0
    for frame in frames:
        name = frame["public_id"].replace("/", "__") + "." + frame.get("format", "jpg")
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            continue
        response = requests.get(frame["secure_url"], timeout=30)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        downloaded += 1

    print(f"{len(frames)} frames in Cloudinary, {downloaded} new downloaded to {RAW_DIR}")


if __name__ == "__main__":
    main()
