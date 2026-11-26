# Foggy Driving box benchmark

This benchmark measures YOLOX before tracking. It uses the 33 images in the fine test split, with 257 car, bus, truck, and motorcycle boxes. Predictions match labels one-to-one at IoU 0.50, in descending score order, regardless of vehicle class. Small boxes have area under 1,024 pixels in the source image. This is fixed-cutoff precision and recall.

| YOLOX-S setting | Small boxes found | Recall | Precision | False positives | Mean CPU detector time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full frame, 0.35 | 8/108 | 44.0% | 86.3% | 18 | 204 ms |
| Full frame, 0.20 | 13/108 | 48.6% | 75.3% | 41 | 205 ms |
| Full frame plus 640 px crops, 0.35 | 27/108 | 58.0% | 74.5% | 51 | 1,456 ms |

The 640 px crops found 19 more small vehicles than the full-frame run at the same cutoff. They also added 33 false positives. Tiling stays opt-in. CPU timing uses four OpenCV threads and includes preprocessing and box decoding.

## Calibrating the crop cutoff

I used the 68 images with coarse annotations (`test_extra`) to choose the cutoff and the 33 fine-annotated images (`test`) to score the chosen settings. The precision floor was the full-frame baseline on the coarse split, 155 matches in 185 predictions (83.78%). Among settings that cleared the floor, I ranked by total matches, then small-vehicle matches, then precision.

The sweep used 640 px crops. It tried full-view cutoffs of 0.35, 0.40, 0.45, and 0.50, crop cutoffs of 0.35, 0.45, 0.55, 0.65, and 0.75, crop box area caps of 32², 64², 96² pixels and no cap, and merge NMS IoUs of 0.35 and 0.45. The report lists all 160 combinations and evaluates only the selected settings and baselines on the fine split.

| Split | Settings | Matched | False positives | Small boxes found | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Coarse | Full view 0.35 | 155/218 | 30 | 14/36 | 83.78% | 71.10% |
| Coarse | Full view and crops 0.35 | 173/218 | 62 | 19/36 | 73.62% | 79.36% |
| Coarse | Full view 0.35, crops 0.75 | 161/218 | 30 | 14/36 | 84.29% | 73.85% |
| Coarse | Full view 0.35, crops 0.75, area cap 96² | 161/218 | 29 | 14/36 | 84.74% | 73.85% |
| Fine | Full view 0.35 | 113/257 | 18 | 8/108 | 86.26% | 43.97% |
| Fine | Full view and crops 0.35 | 149/257 | 51 | 27/108 | 74.50% | 57.98% |
| Fine | Full view 0.35, crops 0.75 | 119/257 | 19 | 10/108 | 86.23% | 46.30% |
| Fine | Full view 0.35, crops 0.75, area cap 96² | 118/257 | 18 | 10/108 | 86.76% | 45.91% |

I kept the crop cutoff as an optional CLI setting. The fixed pixel area cap changed just one false positive and one match on the fine split, and its effect depends on camera resolution. The fine split had already been used to explore tiling and both splits come from Foggy Driving. Treat these results as a check on the cutoff choice. A camera-disjoint video set with labeled events is needed to measure how it transfers to the alert pipeline.

To repeat the sweep, omit `--cuda` for OpenCV CPU inference. To score the chosen cutoff with the shipped detector, use the second command.

```sh
python -m benchmarks.foggy_driving.calibrate Foggy_Driving.zip object_detection_yolox_2022nov.onnx --cuda > calibration.json
python -m benchmarks.foggy_driving.evaluate_boxes Foggy_Driving.zip object_detection_yolox_2022nov.onnx --confidence 0.35 --tile-size 640 --tile-confidence 0.75
```

The corresponding video CLI options are `--confidence 0.35 --detection-tile-size 640 --tile-confidence 0.75`.

## NJIT fog video

The original 360 × 240 NJIT fog video still missed its wrong-way event. The table includes the CPU rerun with 160 px tiles and two smaller-tile trials using the CUDA probe. None of the three runs supplied an observation within 12 px of the hand-traced vehicle at the seven sampled times.

| Tile size | Confidence | Observations | Track IDs | Matched events |
| --- | ---: | ---: | ---: | ---: |
| 160 px | 0.08 | 55 | 15 | 0/1 |
| 112 px | 0.08 | 167 | 27 | 0/1 |
| 112 px | 0.03 | 416 | 43 | 0/1 |

## RTX 5090 timing check

The installed OpenCV and ONNX Runtime builds exposed CPU inference. For the GPU timing check, a local PyTorch CUDA probe ran the operations and weights from the same YOLOX-S ONNX graph. OpenCV still handled resizing and box decoding. The maximum absolute difference in raw outputs was under 0.00007 on two checked images.

| Setting at confidence 0.35 | CPU | RTX 5090 | Box counts |
| --- | ---: | ---: | --- |
| Full frame | 204 ms | 13 ms | 113 matches, 18 false positives on both |
| Full frame plus 640 px crops | 1,456 ms | 67 ms | 149 matches, 51 false positives on both |

These are local per-image averages across the 33 images, including preprocessing and decoding. The shipped CLI and Docker image use OpenCV CPU inference. The graph replay was a timing probe for these weights.

With CUDA-enabled PyTorch and `onnx` installed, run the probe from the repo root.

```sh
python -m benchmarks.foggy_driving.cuda_probe Foggy_Driving.zip object_detection_yolox_2022nov.onnx
```

## Reproduce

Download the Foggy Driving archive from Christos Sakaridis, Dengxin Dai, and Luc Van Gool, *Semantic Foggy Scene Understanding with Synthetic Data*, IJCV 2018. The evaluator reads the archive directly. [Dataset and paper](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/#FoggyDriving)

Use the full-precision OpenCV Zoo YOLOX-S weights. [Model and checksum](../nighttime/README.md)

| File | SHA-256 |
| --- | --- |
| `Foggy_Driving.zip` | `49544ff675752efef2821fd51e80d6c8d298783617698b7f87a06f9f44c3fcac` |

```sh
python -m benchmarks.foggy_driving.evaluate_boxes Foggy_Driving.zip object_detection_yolox_2022nov.onnx --confidence 0.35
python -m benchmarks.foggy_driving.evaluate_boxes Foggy_Driving.zip object_detection_yolox_2022nov.onnx --confidence 0.20
python -m benchmarks.foggy_driving.evaluate_boxes Foggy_Driving.zip object_detection_yolox_2022nov.onnx --confidence 0.35 --tile-size 640
```

For the original fog video, use `wrong.scene.json` and the prepared `wrong.avi`. Set `--detection-tile-size` and `--confidence` to the values in the table. Leave the ROI unset. [Source clip and preparation command](../road_validation/README.md)
