# Camera-Based Obstacle Detection for Driver Assistance (KITTI)

**Group 16**: Hadiya Asif (25L-8020), Raiha Noor (26L-8031), Fizza Zaitoon (26L-8029)
Instructor: Dr. Asma Ahmad | TA: Burhan Hadi Butt | FAST NUCES Lahore

We detect cars, pedestrians and cyclists in a single front-camera image, and test whether this is accurate and fast
enough on a laptop GPU for real-time driver warnings.

* **RQ1** How much does training on KITTI improve over a detector pretrained only on general data (COCO)?
* **RQ2** Which model gives the best accuracy / speed balance, and is it real time on a laptop GPU?
* **RQ3** How much does a random image-level split overestimate performance compared with a split by drive?
* **RQ4** How much do fog, rain and low light hurt accuracy, and does weather augmentation help?

## Results

Test set: 1,543 images from 8 drives that no model saw in training or tuning. Metric: mean AP@0.5 of Car,
Pedestrian and Cyclist at KITTI Moderate difficulty. FPS: RTX 3050 Laptop GPU, batch 1.

| Model | mAP | Car | Pedestrian | Cyclist | FPS |
|---|---|---|---|---|---|
| E0 YOLOv8s, COCO weights only | 0.542 | 0.930 | 0.687 | 0.008 | as E2 |
| E2 YOLOv8s, 640 px | 0.716 | 0.965 | 0.744 | 0.441 | 55 |
| E3 Faster R-CNN ResNet-50 FPN | **0.854** | 0.968 | 0.814 | **0.779** | 14 |
| **FINAL** YOLOv8s, 960 px, oversampling, lr 0.002 | 0.799 | 0.967 | 0.797 | 0.631 | **60** |
| E6 FINAL + weather augmentation | 0.804 | 0.968 | 0.834 | 0.611 | 45 |

* **RQ1**: KITTI training raises mAP from 0.542 to 0.799; cyclists go from 0.008 to 0.631.
* **RQ2**: Faster R-CNN is the most accurate but runs at 11-14 FPS. FINAL is 0.055 behind and real time, so it is
  the deployed model.
* **RQ3**: a random split reports 0.964 instead of 0.799 (+0.165), because near-identical frames leak across splits.
* **RQ4**: the strongest fog / rain / low light cut FINAL to 0.44 / 0.40 / 0.23. Weather augmentation (E6) keeps the
  clean score and raises the average over the 9 corrupted test sets from 0.571 to 0.622.

## Contents

| File | What it does |
|---|---|
| `KITTI_Preprocessing_EDA_Starter_(1) (1).ipynb` | Part 1: loading, cleaning, EDA, split design |
| `data_prep/kitti_drive_grouped_split.ipynb` | train / val / test split by drive (published as a Kaggle dataset) |
| `notebooks/00_common.ipynb` | shared code (evaluator, models, sampling, weather effects); writes `kitti_common.py` |
| `notebooks/01_E0_coco_baseline.ipynb` | E0: COCO-pretrained baseline (RQ1) |
| `notebooks/02_E2_yolov8s.ipynb` | E2: YOLOv8s fine-tuned (RQ1, RQ2) |
| `notebooks/03_E3_faster_rcnn.ipynb` | E3: Faster R-CNN (RQ2) |
| `notebooks/04_E4_tuning.ipynb` | E4: tuning on validation (image size, oversampling, learning rate) |
| `notebooks/05_FINAL.ipynb` | FINAL: chosen settings, tested once, error analysis |
| `notebooks/06_E5_random_split.ipynb` | E5: same recipe on a random split (RQ3) |
| `notebooks/07_E6_weather.ipynb` | E6 + E8: weather-augmented training and fog / rain / low-light tests (RQ4) |
| `notebooks/08_speed_and_app.ipynb` | E7: laptop GPU speed, the pipeline and the demo app (RQ2) |
| `notebooks/09_results_and_RQs.ipynb` | all results together and the answers to RQ1-RQ4 |
| `app/` | demo app (`app.py`) and pipeline (`pipeline.py`) |
| `results/` | metrics, predictions and figures of every experiment; `experiments.csv` lists them all |
| `weights/` | `FINAL.pt` and `E6.pt` (the app's models); the other weights are in the GitHub Release |

## Data

[KITTI 2D Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) (Geiger et al.,
CVPR 2012): 7,481 labelled images from 141 drives, licence CC BY-NC-SA 3.0. We split it by drive into train / validation /
test (4,699 / 1,239 / 1,543 images), so near-identical frames never appear in two splits. The split is published as
[`fizzazaitoonbsds2022/kitti-drive-grouped-split`](https://www.kaggle.com/datasets/fizzazaitoonbsds2022/kitti-drive-grouped-split).
The data is not in this repository.

## Setup

```bash
conda create -n kitti python=3.11 -y
conda activate kitti
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Reproduce

* **Notebooks 00-07 ran on Kaggle** (T4 GPU, about 10 GPU-hours). Attach the split dataset and the output of 00, then
  *Run All*. Notebooks 05-07 also need the outputs of 04 (and 06-07 those of 05). `SMOKE = True` gives a quick test run.
* **Notebooks 08 and 09 run locally** from `notebooks/`. Notebook 09 only needs `results/`. Notebook 08 needs a CUDA GPU,
  the weights `E2.pt`, `E3.pt`, `FINAL.pt`, `E6.pt`, and the test split in `data/` (see below).

## Demo app

```bash
python app/app.py        # then open http://127.0.0.1:7860
```

Upload a front-camera road image or click an example. The app shows the detected obstacles with a rough distance,
our lane, a CLEAR / CAUTION / DANGER warning and the time per image. The model can be switched between FINAL and E6.
It runs fully offline. Distances assume a KITTI-like camera, so they are only meaningful for similar images.

The example images in `app/examples/` come from KITTI test drives (CC BY-NC-SA 3.0, Geiger et al.), plus fog and
low-light copies of one of them.
