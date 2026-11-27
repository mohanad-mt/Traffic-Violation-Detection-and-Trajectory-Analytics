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

## RF-DETR Small

I ran the released 512 px COCO checkpoint on the same 300 DAWN fog images and replayed its boxes through the existing tracker and event rules. The benchmark used FP32 PyTorch on the RTX 5090. The checkpoint and package version are pinned. Detector timings include preprocessing and box conversion, with a five-call warm-up and no video decode. The reference timings are from the earlier run on the same machine. [Model and weights](https://github.com/roboflow/rf-detr) · [Run details](runs/rfdetr-small.json)

| Detector | DAWN AP50–95 | AP50 | Small AP | Small AR | Mean | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOX-M | 54.18 | 82.92 | 26.66 | 39.93 | 12.4 ms | 13.8 ms |
| RT-DETR-R18 | 49.26 | 71.59 | 21.81 | 38.81 | 20.2 ms | 27.6 ms |
| RF-DETR-S | 55.83 | 84.39 | 26.54 | 43.93 | 12.2 ms | 16.9 ms |

At a score cutoff of 0.50, RF-DETR-S matched 1,186 boxes with 68 false positives at IoU 0.50. YOLOX-M matched 1,159 with 64 false positives at that cutoff. Both matched 66 of the 270 small boxes. Lowering RF-DETR-S to 0.20 found 380 more boxes and added 993 false positives. This is a box-level comparison on DAWN. I evaluated the event thresholds separately using footage.

| Development replay | TP | FP | FN | F1 | Matched |
| --- | ---: | ---: | ---: | ---: | --- |
| RT-DETR-R18, OC-SORT, 0.20 | 2 | 0 | 2 | 0.67 | Stop, red |
| RF-DETR-S, OC-SORT, 0.20 | 2 | 0 | 2 | 0.67 | Stop, red |
| RF-DETR-S, ByteTrack, 0.20 start / 0.03 continuation | 3 | 1 | 1 | 0.75 | Fog, stop, red |

In the fog clip, RF-DETR-S put a box within 12 px of the reference at seven of seven sampled times above 0.03, with two above 0.20. The ByteTrack run confirmed the wrong-way event at 3.73 s. The fixed OC-SORT setting missed that event with both detectors. Applying a lane crop with 24 px of padding to RF-DETR-S and ByteTrack missed both the fog and shadow references. The full-frame ByteTrack and OC-SORT settings also missed both nighttime transfer events and triggered zero alerts on those two short clips. Across the four sampled positions per nighttime clip, the strongest nearby RF-DETR scores were 0.198 on Camera 1 and 0.089 on Camera 2. [Nighttime transfer footage](night_transfer/README.md)

The extra stop alert came from the same ID as the matched alert. The vehicle's bottom-center estimate drifted roughly 4 px from the beginning of the first stationary window, which the default 4 px radius initially accepted. Replaying the same boxes at a 2 px radius matched the stop once, with an estimated onset of 4.27 s and no extra alert. Three intermediate radii produced onsets outside the fixed tolerance. This sweep is camera-specific, and there is only one positive stop clip.

RF-DETR-S is available through `--detector rfdetr`. The model recovered the fog alert on development footage, and the stop and nighttime results require more camera-specific evidence before choosing a general default.

### Reproduce the RF-DETR run

Follow the [CUDA and RF-DETR setup](../README.md#cuda-setup) and prepare the road and night clips below. Install `pycocotools==2.0.11` in the same environment for image scoring.

```sh
python -m benchmarks.assets build/assets dawn-fog rfdetr-small
python -m benchmarks.compare_detectors dawn-rfdetr-small build/assets/DAWN-Fog.zip build/assets/rf-detr-small.pth --dataset dawn-fog --device cuda --detector rfdetr
python -m benchmarks.road_validation.compare build/assets/rf-detr-small.pth build/road build/nighttime/sample.avi --device cuda --detector rfdetr --tracker point bytetrack ocsort --cache-dir build/road/boxes-rfdetr
```

To repeat the transfer check, use the [nighttime commands](night_transfer/README.md) with the RF-DETR checkpoint and `--detector rfdetr --confidence .20`. Run once with `--tracker ocsort` and once with `--tracker bytetrack --continuation-confidence .03`. For the development lane crop, use `--detection-roi 74 103 218 240` on the fog clip and `--detection-roi 34 72 256 240` on the shadow clip. The stop radius uses the `stopped_radius_px` scene threshold.

### Point tracking from a known sighting

I tested whether a point tracker could follow a vehicle once its position was known. TAPNext++ processed frames in order from the first labeled sighting, with a single query on the bottom-center annotation or 3 px above it. For the raised seed, I added the same offset back to compare predictions with the bottom-center references. A hit requires a visible prediction within 12 px at a later labeled time. The red sedan is the visible control. The stationary baseline keeps the original labeled position. [TAPNext++ release](https://github.com/google-deepmind/tapnet/tree/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnextpp/votsp2026)

| Clip | Later reference points | Stationary baseline | Seed at bottom center | Seed 3 px above |
| --- | ---: | ---: | ---: | ---: |
| Fog | 6 | 1 | 1 | 1 |
| Red | 5 | 0 | 5 | 5 |
| Night camera 1 | 3 | 0 | 0 | 0 |
| Night camera 2 | 3 | 0 | 0 | 3 |

On Camera 2, the raised seed followed the vehicle to within 0.7–7.8 px at the three checkpoints. Both seeds stayed near their initial position on Camera 1 even though the model marked them visible. The later frames averaged about 33 ms each on the RTX 5090. This was a diagnostic with a hand-labeled first sighting. An automatic trial needs a reliable seed from the detector, new raw camera views with vehicle trajectories, and longer negative clips to measure how often the tracker follows the wrong object.

Install `einops==0.8.0` in the CUDA PyTorch environment. Use the pinned source checkout and put its root on `PYTHONPATH`. The optional checkpoint is 2.53 GB.

```sh
git clone https://github.com/google-deepmind/tapnet.git build/tapnet
git -C build/tapnet checkout ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b
python -m benchmarks.assets build/assets tapnextpp-512
PYTHONPATH=build/tapnet python -m benchmarks.road_validation.point_propagation build/assets/tapnextpp_512.ckpt build/road build/independent
```

## External references and pipeline revisions

I compared the 704 px RF-DETR-L checkpoint with Small and held ByteTrack, its 0.20 start and 0.03 continuation cutoffs, and the event scenes fixed. Large improved DAWN AP50–95 from 55.83 to 57.44 and small-object AP from 26.54 to 28.68. On the road clips the same configuration matched one event with Large and three with Small. Large had three extra alerts. Both used FP32 on the RTX 5090. [Run record](runs/research-revision.json)

| Development run | TP | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: |
| RF-DETR-S, ByteTrack, existing rules | 3 | 1 | 1 | 0.75 |
| RF-DETR-L, ByteTrack, existing rules | 1 | 3 | 3 | 0.25 |
| RF-DETR-S, ByteTrack, 0.75 px/s stop drift bound | 3 | 0 | 1 | 0.86 |

The stop check limits net displacement divided by elapsed time when an alert is about to confirm. At the existing 4 px anchor radius, 0.60, 0.75, and 0.90 px/s each removed the early stop alert on this clip. The matched stop kept its 4.67 s estimated onset and 7.73 s confirmation. Batch and streaming emitted that same event from the video. The result is a one-camera development trial. A separate direction prototype used the latest point and a point roughly 0.25, 0.40 or 0.60 s earlier to reduce per-frame jitter. It recovered no extra match and added false alerts, so the application uses the original direction rule.

On the two nighttime transfer clips, Large with the fixed ByteTrack setting missed both events and added one false alert. Lowering the start cutoff to 0.10 after looking at the misses recovered Camera 1 for each model and added three false alerts per model across both clips. This is an exploratory threshold check on previously examined footage.

### FETV footage and clock accuracy

The official download contained 200 clips from nine source prefixes. I inspected all video packet timestamps and checked the worst clip using decoded presentation timestamps. In 198 clips the packet timestamp intervals varied. The maximum absolute difference between the frame-index clock and the packet clock exceeded 100 ms on 20 clips and reached 142 ms on one. The median of the per-clip maxima was 66 ms. Timing differences of this size can move a crossing across a signal transition.

Both the batch and streaming paths accept `--timestamp-source pts`. The renderer uses the same clock when the prediction JSON identifies that mode. A generated variable-rate test verifies frame timing through tracking, alert confirmation and MP4 export. The stop-clip overlay preserved all 158 timestamps to within one microsecond. The FETV archive contains no ground-truth JSON, so these are timing and access results. [Dataset release](https://github.com/MoyoG/FETV) · [Missing annotation issue](https://github.com/MoyoG/FETV/issues/1)

### Reproduce these revisions

```sh
python -m benchmarks.assets build/assets dawn-fog rfdetr-small rfdetr-large fetv-clips
python -m zipfile -e build/assets/FETV_public_clips.zip build/fetv
python -m benchmarks.video_clock build/fetv/FETV_public_clips
python -m benchmarks.compare_detectors dawn-rfdetr-large build/assets/DAWN-Fog.zip build/assets/rf-detr-large-2026.pth --dataset dawn-fog --device cuda --detector rfdetr-large
python -m benchmarks.road_validation.compare build/assets/rf-detr-large-2026.pth build/road build/nighttime/sample.avi --device cuda --detector rfdetr-large --tracker bytetrack
python -m benchmarks.road_validation.compare build/assets/rf-detr-small.pth build/road build/nighttime/sample.avi --device cuda --detector rfdetr --tracker bytetrack --max-stopped-drift-px-s .75
```

For the stop-clip confirmation check, set `max_stopped_drift_px_s` to 0.75 in a copy of its scene. Run the main video CLI with RF-DETR-S, ByteTrack and the same cutoffs. Use `--timestamp-source pts --include-tracks` for batch output and `--timestamp-source pts --stream-events` for incremental output. Pass the batch JSON to `traffic_trajectory.visualize` to render the PTS overlay.

### Continuous stopped-vehicle check

I retrieved video 2 and the training timestamps from the official AI City 2021 Track 4 archive. The release labels a stop onset at 587 s. I drew two roadway polygons on an earlier frame at 30 s and ran RF-DETR-S and ByteTrack at 5 FPS through the full 894 s video. Each rule setting used the same tracks. The score matches one onset within 10 s and counts other alerts as unmatched. This is a time-only diagnostic on one public training video, because the official label supplies neither a vehicle ID nor a lane. [Download](https://www.aicitychallenge.org/2021-track4-download/) · [Label and evaluation format](https://www.aicitychallenge.org/2021-data-and-evaluation/)

| Stop duration | Drift limit | Unmatched alerts | Time matches | Delay from labeled onset |
| --- | --- | ---: | ---: | ---: |
| 3 s | Off | 749 | 1 | 2.4 s |
| 3 s | 0.75 px/s | 585 | 1 | 2.4 s |
| 10 s | Off or 0.75 px/s | 38 | 1 | 14.4 s |
| 20 s | Off or 0.75 px/s | 6 | 0 | — |

At 3 s with the drift bound, 401 of 585 unmatched alerts began within the top 50 px, near the highway horizon. The time match came from a 7 × 6 px box in that region. The 10 s match came from a different track lower in the frame. The time matches alone cannot establish that either setting followed the incident vehicle. The clip shows that fixed pixel thresholds and short dwell can generate many candidates even with a drift limit. The latency and missed match at longer durations argue for camera-aware motion or other stop evidence before choosing a new global setting. [Run record](runs/research-revision.json)

```sh
python -m benchmarks.road_validation.aicity21_stop build/aicity21-track4-train-data/2.mp4 build/assets/rf-detr-small.pth --device cuda
```

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
