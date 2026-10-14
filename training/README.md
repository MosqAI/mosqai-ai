# Training the mosquito model

The server starts with the rules detector (`DETECTOR=rules`), which only checks
what is new in the tube and its size, shape and contrast. These steps train a YOLOv8 model that has actually seen
mosquitoes, using the photos your traps collect. `data/` and `runs/` are git-ignored.

## 1. Collect photos

Set `COLLECT_MODE=true` in `.env` and restart the server. Every frame is uploaded
to Cloudinary under `mosqai_dataset/trap_<id>/`. Aim for **300+ photos with
mosquitoes** plus a good number of empty or fly/debris photos, under the lighting
the traps really have.

```bash
pip install -r training/requirements.txt
python training/download_dataset.py           # -> training/data/raw/
```

## 2. Label

Draw a box around every mosquito with one class, `mosquito`. Free tools:
[Label Studio](https://labelstud.io), [CVAT](https://www.cvat.ai) or
[Roboflow](https://roboflow.com). Export in **YOLO** format and unpack it so you have:

```
training/data/labelled/images/*.jpg
training/data/labelled/labels/*.txt
```

Box only mosquitoes, not flies or other insects: those are what the model must learn to ignore.

## 3. Train

```bash
python training/prepare_dataset.py            # 80/20 train/val split + data.yaml
python training/train.py                      # add --device 0 on a GPU machine
```

On a CPU this takes hours; a free Google Colab GPU takes about 20-60 minutes. On
Colab, upload the `mosqai-ai` folder (with `training/data/labelled/`), run the two
commands above with `--device 0`, and download `models/yolov8n_mosquito.onnx`.

Training prints **precision** (how many detections were real mosquitoes) and
**recall** (how many mosquitoes were found). Plots and the best weights are in
`training/runs/mosquito*/`. Below ~0.8 on either, label more photos and retrain.

## 4. Switch the server to the model

In `.env`:

```
DETECTOR=ensemble                # the model, confirmed by the rules (or yolo for the model alone)
MODEL_VERSION=mosqai-v2          # bump on every new model
TARGET_CLASS_IDS=                # one class, so count all
```

`ensemble` is the safer choice: a model trained on a few hundred photos still
mistakes some glare for mosquitoes, and the rules (with the empty-tube photo)
reject anything that isn't really new in the tube.

Restart (`docker compose up -d`) and check `GET /api/v1/health` shows the new
`model_version`. Keep collect mode on for a while to gather more training data,
then turn it off.
