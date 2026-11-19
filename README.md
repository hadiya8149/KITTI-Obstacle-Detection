# Camera-Based Obstacle Detection for Driver Assistance (KITTI)

**Group 16**: Hadiya Asif (25L-8020), Raiha Noor (26L-8031), Fizza Zaitoon (26L-8029)
Instructor: Dr. Asma Ahmad | TA: Burhan Hadi Butt | FAST NUCES Lahore

We detect cars, pedestrians, cyclists and other road obstacles in a single front-camera image, and ask whether this
is accurate enough, and fast enough on a laptop GPU, to drive real-time driver warnings. The project answers four
research questions:

* **RQ1** How much does training on KITTI improve detection over a detector pretrained only on general-purpose data?
* **RQ2** Which approach gives the best balance between accuracy and speed, and can it run in real time on a laptop GPU?
* **RQ3** How much does a random image-level split overestimate performance compared with a split by drive?
* **RQ4** How much does accuracy drop under simulated fog, rain and low light, and can training augmentation reduce the drop?

## Contents

| Part | File | Covers |
|---|---|---|
| 1 | `KITTI_Preprocessing_EDA_Starter_(1) (1).ipynb` | dataset loading and description, validation and cleaning, difficulty labels, drive-grouped split design, EDA, YOLO export |
| 2 | `data_prep/kitti_drive_grouped_split.ipynb` | step 1 of Part 2: the train / validation / test split by drive, packaged as a Kaggle dataset |
| 2 | `notebooks/00_common.ipynb` | shared code with explanations: data access, KITTI evaluator, detectors, class-imbalance sampling, weather corruptions, training loops, plots |
| 2 | `notebooks/01` ... `07` | one notebook per experiment, each run end to end on Kaggle with its outputs kept (see the table below) |
| 2 | `notebooks/08_speed_and_app.ipynb` | E7 speed benchmark on the laptop GPU, the final pipeline and the demo app |
| 2 | `notebooks/09_results_and_RQs.ipynb` | all results side by side, answers to RQ1-RQ4, conclusions and limitations |

| Notebook | Experiment | Answers |
|---|---|---|
| 01 | E0: COCO-pretrained YOLOv8s, no KITTI training (baseline) | RQ1 |
| 02 | E2: YOLOv8s fine-tuned on KITTI | RQ1, RQ2 |
| 03 | E3: Faster R-CNN (ResNet-50 FPN) fine-tuned on KITTI | RQ2 |
| 04 | E4: tuning YOLOv8s (image size, class-imbalance sampling, learning rate) on validation | model choice |
| 05 | FINAL: the chosen settings, trained in full, tested once, error analysis | all |
| 06 | E5: the FINAL recipe on a random image-level split | RQ3 |
| 07 | E6 + E8: weather-augmented training; fog / rain / low-light test sets | RQ4 |
| 08 | E7: speed on the laptop GPU; pipeline check; demo | RQ2 |
| 09 | all results and answers | all |

Every notebook has Markdown explanations and an **Observations** section with the real numbers.

## Repository layout

```
KITTI_Preprocessing_EDA_Starter_(1) (1).ipynb   Part 1 (data and EDA)
data_prep/kitti_drive_grouped_split.ipynb       Part 2 step 1: the split (run once, on Kaggle, CPU)
notebooks/00_common.ipynb ... 09_results_and_RQs.ipynb
notebooks/kitti_common.py                        shared module, written by 00_common.ipynb (do not edit by hand)
app/pipeline.py                                  final pipeline: image -> obstacles, distance, warning (written by notebook 08)
app/app.py                                       Gradio demo app (written by notebook 08)
app/examples/                                    test-drive images (and fog / low-light copies) for the demo
results/<experiment>/                            record.json (settings + metrics), history.csv, per-class tables,
                                                 predictions (.parquet) and figures/ of every experiment
results/experiments.csv                          the experiment log: one row per experiment (built by notebook 09)
weights/FINAL.pt, weights/E6.pt                  the two models the app uses (all weights: GitHub Release)
```

## Dataset

[KITTI 2D Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
(Geiger et al., CVPR 2012): 7,481 labelled images (about 1242 x 375 px) from 141 recording drives, 40,570 objects in
8 classes, plus 11,295 `DontCare` regions. Licence: CC BY-NC-SA 3.0 (non-commercial academic use).

We read it from the Kaggle mirror [`klemenko/kitti-dataset`](https://www.kaggle.com/datasets/klemenko/kitti-dataset)
and publish the split version as
[`fizzazaitoonbsds2022/kitti-drive-grouped-split`](https://www.kaggle.com/datasets/fizzazaitoonbsds2022/kitti-drive-grouped-split):

| Split | Images | Drives | Car | Pedestrian | Cyclist |
|---|---|---|---|---|---|
| train | 4,699 | 119 | 18,169 | 3,165 | 1,050 |
| validation | 1,239 | 14 | 5,081 | 658 | 322 |
| test | 1,543 | 8 | 5,492 | 664 | 255 |

Whole drives go to one split, so near-identical neighbouring frames never appear on both sides. A random image-level
split is kept only to measure that leakage (RQ3).

## Setup

```bash
conda create -n kitti python=3.11 -y
conda activate kitti
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build of PyTorch
pip install -r requirements.txt
```

## How to reproduce

**Training and evaluation (notebooks 01-07) ran on Kaggle** (Tesla T4, 2 sessions in parallel, about 10 GPU-hours
in total), because the data is 12 GB and the laptop GPU has 4 GB.

1. Run `data_prep/kitti_drive_grouped_split.ipynb` on Kaggle with `klemenko/kitti-dataset` attached (CPU, about
   15 minutes), and save its output as a dataset. Or simply use our published dataset (link above).
2. Upload `notebooks/00_common.ipynb` as a Kaggle notebook, attach the split dataset and run it. Its output is
   `kitti_common.py`.
3. For each of 01-07: create a Kaggle notebook from the file, enable GPU (T4) and internet, attach the split dataset
   and the output of 00. Notebooks 05-07 also need the outputs of the notebooks they build on: 05 needs 04; 06 needs
   04 and 05; 07 needs 04 and 05. Then *Run All*. Each notebook installs the pinned library versions itself.
   Set `SMOKE = True` in the first code cell for a quick end-to-end test on 5 % of the data.
4. Download each notebook's output (`results/`, `weights/`) into this repository.

**Notebooks 08 and 09 run locally**, from the `notebooks/` folder, with the results already in `results/`:

* 09 needs no GPU and no images.
* 08 needs a CUDA GPU, the weights `E2.pt`, `E3.pt`, `FINAL.pt`, `E6.pt` in `weights/`, and the test split in
  `data/`. Download `images_test.zip`, `labels_yolo.zip`, `tables/` and `splits/` from the split dataset into
  `data/`. Run it with the laptop plugged in and Windows set to *Best performance*.

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 08_speed_and_app.ipynb
```

## Demo app

```bash
python app/app.py        # then open http://127.0.0.1:7860
```

Upload a front-camera road image (or click an example). The app draws the detected obstacles with a rough distance
estimate, highlights our lane, and shows a CLEAR / CAUTION / DANGER warning and the time per image. The model can be
switched between FINAL (default) and the weather-augmented E6. It runs fully offline.

## Results

Test set: 1,543 images from 8 drives that no model saw in training or model selection. Primary metric = mean AP@0.5
of Car, Pedestrian and Cyclist at KITTI Moderate difficulty (40-point interpolated AP, `DontCare` regions ignored).
FPS: RTX 3050 Laptop GPU, batch 1, including pre-processing and NMS (notebook 08).

| Model | Primary | Car | Pedestrian | Cyclist | FPS |
|---|---|---|---|---|---|
| E0 YOLOv8s, COCO weights only | 0.542 | 0.930 | 0.687 | 0.008 | as E2 |
| E2 YOLOv8s, 640 px | 0.716 | 0.965 | 0.744 | 0.441 | 55 |
| E3 Faster R-CNN ResNet-50 FPN | **0.854** | 0.968 | 0.814 | **0.779** | 14 |
| **FINAL** YOLOv8s, 960 px, oversampling, lr 0.002 | 0.799 | 0.967 | 0.797 | 0.631 | **60** |
| E6 FINAL + weather augmentation | 0.804 | 0.968 | 0.834 | 0.611 | 45 |

* **RQ1**: training on KITTI raises the primary metric from 0.542 to 0.799 (+0.257); cyclists go from 0.008 to 0.631.
* **RQ2**: Faster R-CNN is the most accurate but runs at 11-14 FPS. The tuned YOLOv8s (FINAL) is 0.055 behind and
  real time on a laptop GPU, so it is the deployed model.
* **RQ3**: a random image-level split reports 0.964 instead of 0.799 (+0.165).
* **RQ4**: at the strongest severity, fog / rain / low light cut FINAL to 0.44 / 0.40 / 0.23. Weather augmentation (E6)
  keeps the clean score and raises the average over the 9 corrupted test sets from 0.571 to 0.622 (low light,
  severity 3: 0.226 to 0.405).

The images in `app/examples/` are from KITTI test drives (CC BY-NC-SA 3.0, Geiger et al.), included for the demo.
