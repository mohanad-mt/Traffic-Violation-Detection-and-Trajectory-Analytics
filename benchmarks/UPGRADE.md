# Detector, tracking and event validation

The reference pipeline is the [crop-calibration commit](https://github.com/mohanad-mt/Traffic-Violation-Detection-and-Trajectory-Analytics/commit/3c854f4581e3c1ae10451cd21228842b85d90e4b). It uses YOLOX-S, OpenCV CPU inference, greedy point association and event matching by lane, type and onset. The 31 original tests passed before these changes.

## Detector comparison

I compared the pretrained checkpoints without fine-tuning. The 33 fine Foggy Driving images are development data because they had already been used to inspect tiling. DAWN V1 Fog provided an external set of 300 images and 1,818 labeled cars, buses, trucks and motorcycles. Its source metadata is insufficient to establish independent camera IDs. The DAWN comparison used each fixed detector configuration once. The cutoff rows show trade-offs from cached predictions.

| Detector | Foggy Driving AP50–95 | DAWN AP50–95 | DAWN AP50 | DAWN AP75 | DAWN small AP | DAWN mean detector time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOX-S | 32.43 | 47.12 | 72.99 | 49.36 | 22.54 | 7.9 ms |
| YOLOX-M | 41.66 | 54.18 | 82.92 | 56.60 | 26.66 | 12.4 ms |
| RT-DETR-R18 | 41.04 | 49.26 | 71.59 | 52.60 | 21.81 | 20.2 ms |
| YOLOX-S with 640 px crops at 0.75 | See the crop cutoff sweep | 46.43 | 73.16 | 47.42 | 22.52 | 47.1 ms |

The table pools the four vehicle classes. The report also includes class-aware AP, size-specific average recall, and per-cutoff precision, recall, F1 and counts. AP uses detections above 0.01 after the model's standard postprocessing. The tiled run applies the stricter crop cutoff before merging. YOLOX uses class-agnostic NMS. RT-DETR supplies its own ranked box proposals. DAWN boxes use source-image coordinates and XML extents. Times are synchronized per-image means on the RTX 5090 after warm-up and include preprocessing and decoding boxes.

At confidence 0.35 on DAWN, YOLOX-M found 1,279 vehicles with 103 false positives. YOLOX-S found 1,073 with 91. The calibrated crop pass found eight more labels than YOLOX-S at that cutoff and added four false positives. Its ranked AP and runtime make tiling a poor general default for this dataset.

## Tracking comparison

GRAM-RTM supplies ground-truth track IDs and the authors' detection files. I mapped the one-based detection frame numbers to the zero-based XML indices and scored both sequences with TrackEval. All supplied boxes went in as strong detections so the benchmark isolates association. The authors' scores use a different scale from the neural models. Low-score continuation was exercised separately on the video replays below.

| Sequence | Tracker | HOTA | IDF1 | MOTA | ID switches | Fragments |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M-30 | Point | 31.5 | 55.5 | 35.4 | 137 | 905 |
| M-30 | ByteTrack | 32.0 | 56.8 | 36.1 | 2 | 875 |
| M-30 | OC-SORT | 30.1 | 54.6 | 34.9 | 3 | 589 |
| Urban1 | Point | 19.9 | 18.7 | 41.8 | 369 | 1,164 |
| Urban1 | ByteTrack | 24.6 | 24.4 | 42.2 | 117 | 960 |
| Urban1 | OC-SORT | 24.6 | 23.0 | 40.7 | 65 | 368 |

ByteTrack had the highest IDF1 on both sequences. OC-SORT yielded fewer fragments, with lower MOTA than the point tracker on both. This is an association benchmark on two sequences and a common legacy detector. The point tracker remains the default to preserve the baseline and the lightweight installation.

## End-to-end alerts

The development replay covers 384 frames across five clips, including 8 s of negative nighttime coverage. Detection ran once per model and clip, with boxes cached at 0.03. The replay evaluated the point tracker, ByteTrack and OC-SORT at track-start scores 0.08, 0.20 and 0.35, with either the same cutoff for continuation or a 0.03 floor. OC-SORT itself associates only high-score detections. The red sedan and fog case use a spatial reference. The highway stop and shadow labels only verify time. Onset tolerances were fixed at 0.5 s for the red clip and 1.0 s for the other positive clips.

| Development setting | TP | FP | FN | F1 | Matched cases |
| --- | ---: | ---: | ---: | ---: | --- |
| YOLOX-S, point, 0.35 | 1 | 0 | 3 | 0.40 | Red |
| YOLOX-S, ByteTrack, 0.08 | 2 | 1 | 2 | 0.57 | Stop, red |
| YOLOX-M, point, 0.35 | 1 | 0 | 3 | 0.40 | Red |
| RT-DETR-R18, OC-SORT, 0.20 | 2 | 0 | 2 | 0.67 | Stop, red |

I froze the original setting and the best development setting before running the two new nighttime camera clips. Both matched zero of two wrong-way events and produced zero alerts. Precision is undefined on those clips. The published recordings contain their original detector overlays and the reference trajectories are approximate. Two missed targets are enough to show that the development gain did not transfer here. The result cannot establish an operational false-alert rate. [Clips, labels and commands](night_transfer/README.md)

On the original NJIT fog clip, no YOLOX-S or YOLOX-M box at the 0.03 cache floor landed within 12 px of the traced vehicle at any of the seven sampled times. RT-DETR had nearby boxes at four of those times at 0.03 and at none at 0.20. This locates a problem before association. The trial did not establish whether additional training or better camera imagery would recover a reliable series of boxes.

## CUDA parity

The actual video executable uses ONNX Runtime CUDA for YOLOX. With FP32 and TF32 disabled, the largest raw output difference on the checked red clip frame was 0.000064. Both backends produced 294 observations, 21 tracks and the same red crossing within 1 ms. On the 33-frame 480 × 270 clip, the mean detector time was 242.4 ms on OpenCV CPU and 8.5 ms on the RTX 5090. End-to-end processing was 4.1 and 111.4 FPS after warm-up. The rates exclude model loading and are local short-clip measurements.

## Reproduce

Install the optional packages described in the [README](../README.md#detectors-and-trackers). Obtain the pinned weights and evaluation inputs.

```sh
python -m benchmarks.assets build/assets dawn-fog gram rtdetr-config rtdetr-processor rtdetr-weights
python -m benchmarks.compare_detectors dawn-s build/assets/DAWN-Fog.zip object_detection_yolox_2022nov.onnx --dataset dawn-fog --device cuda
python -m benchmarks.compare_detectors dawn-m build/assets/DAWN-Fog.zip yolox_m.onnx --dataset dawn-fog --device cuda --input-color bgr
python -m benchmarks.compare_detectors dawn-r18 build/assets/DAWN-Fog.zip build/assets/rtdetr-r18 --dataset dawn-fog --device cuda --detector rtdetr
python -m benchmarks.compare_detectors dawn-tiles build/assets/DAWN-Fog.zip object_detection_yolox_2022nov.onnx --dataset dawn-fog --device cuda --tile-size 640 --tile-confidence .75
python -m benchmarks.gram_tracking build/assets/GRAM-RTMv4.zip
python -m benchmarks.backend_parity object_detection_yolox_2022nov.onnx build/road/red.avi benchmarks/road_validation/red.scene.json
```

Use `--dataset foggy-driving` with the Foggy Driving archive to repeat the development detector comparison. For the replay, prepare `wrong.avi`, `stop.avi`, `red.avi`, and the shadow clip as `second.avi` in the same folder. Prepare the night negative with the nighttime script. Run the replay for each model, with `--detector rtdetr` for RT-DETR and `--input-color bgr` for the upstream YOLOX-M export. [Road clip preparation](road_validation/README.md) · [Night negative](nighttime/README.md) · [Upstream YOLOX checkpoints](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md)

```sh
python -m benchmarks.road_validation.compare object_detection_yolox_2022nov.onnx build/road build/nighttime/sample.avi --device cuda --tracker point bytetrack ocsort --cache-dir build/road/boxes-s
```

The [run records](runs/upgrade.json) retain exact scores, counts and inputs. The [research notes](RESEARCH.md) record the alternatives and data access issues. A follow-up compared [same-frame and prior-frame crop guidance](road_validation/TRACK_GUIDANCE.md) on the fog clip.
