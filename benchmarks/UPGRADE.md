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

### Published stall detector comparison

Zhao et al. took first place on AI City 2021 Track 4 with 0.9524 F1, 5.3080 s onset RMSE and a 0.9355 composite score. Track 4 covers timed stalls and crashes on fixed cameras. The official code index marks the top three entries private. Team 76's public pointer has no runnable source or weights, and the third-place public repository omits the anomaly detection stages described in its README. The 2020 winner publishes its stages and training recipes, which require a separately trained detector and a legacy Paddle stack. [2021 paper](https://openaccess.thecvf.com/content/CVPR2021W/AICity/papers/Zhao_Good_Practices_and_a_Strong_Baseline_for_Traffic_Anomaly_Detection_CVPRW_2021_paper.pdf) · [Code index](https://github.com/NVIDIAAICITYCHALLENGE/2021AICITY_Code_From_Top_Teams) · [Team 76 pointer](https://github.com/Endeavour10020/AiCity-2021-Track4) · [Team 92 release](https://github.com/jingyuanchan/CVPR_AICity_2021_Anomaly_Detection) · [2020 winner](https://github.com/Cury30/AICity2020-Anomaly-Detection)

The replay implements the winner's core stall-detection idea. It runs a MOG2 background model in each direction at 5 Hz, detects vehicles in each background at about 1 Hz, and groups reverse-pass boxes by overlap with a fixed anchor. A candidate needs 30 s of persistence, at least 60% detection support and an overlapping forward-pass detection. An appearance check looks back up to 30 s in the original frames, using the paper's 20 dB PSNR threshold, and spatial deduplication keeps the longer tube. The 4 s background history and variance threshold of 16 come from the released 2020 code. This implementation uses the project's RF-DETR-S checkpoint and fixed road polygons. The published system used a trained detector ensemble, automatic road masks, camera stabilization, pixel-state matrices, SSIM tube filtering and collision-onset refinement.

Run both detectors on the original training clips. Video 2 has one labeled incident at 587 s. Video 1 is a congestion control with no incident in the release's annotation file. Both methods use the same per-clip road polygons, PTS clock, RF-DETR-S checkpoint and one-to-one onset scoring within ±10 s. These are local, time-only scores on two training clips. Reverse modeling uses future frames, so the tube support timestamp cannot be used as live alert latency. The official test composite above is for the complete published system on the separate test set. [Download](https://www.aicitychallenge.org/2021-track4-download/) · [Label and evaluation format](https://www.aicitychallenge.org/2021-data-and-evaluation/)


| Detector | Video 2 time matches | Video 2 unmatched alerts | Video 1 alerts |
| --- | ---: | ---: | ---: |
| RF-DETR-S + ByteTrack, 3 s default | 1 | 749 | 29 |
| RF-DETR-S + ByteTrack, 3 s and 0.75 px/s drift limit | 1 | 585 | 19 |
| RF-DETR-S + ByteTrack, 10 s | 1 | 38 | 0 |
| RF-DETR-S + ByteTrack, 20 s | 0 | 6 | 0 |
| Bidirectional background replay | 1 | 7 | 0 |

The background replay estimated 591.2 s for the labeled 587 s incident. Its matched box overlaps a vehicle that is already stationary at 587 s and is still there at 820 s. The 3 s match came from a different position near the highway horizon. The 10 s rule matched the same image region as the background replay at 591.4 s. Manual spot checks of the two regions found the eight background alerts clustered around two stationary vehicles. Gaps and variation in box overlap can split one stationary vehicle into several tubes. The public labels supply one timestamp for the incident and give no vehicle identity, so the table counts all additional alerts as unmatched.

Use the background replay as the offline stall baseline. Evaluate a past-only version with track-gap recovery on additional cameras before replacing the live stop rule. Measure confirmation delay and false alerts on congestion, and use vehicle-level labels to verify which stalled vehicle triggered each alert. [Run record](runs/aicity21-background.json)

```sh
python -m benchmarks.road_validation.aicity21_stop build/aicity21-track4-train-data/2.mp4 build/assets/rf-detr-small.pth --clip 2 --device cuda
python -m benchmarks.road_validation.aicity21_background build/aicity21-track4-train-data/2.mp4 build/assets/rf-detr-small.pth --clip 2 --device cuda
python -m benchmarks.road_validation.aicity21_stop build/aicity21-track4-train-data/1.mp4 build/assets/rf-detr-small.pth --clip 1 --device cuda
python -m benchmarks.road_validation.aicity21_background build/aicity21-track4-train-data/1.mp4 build/assets/rf-detr-small.pth --clip 1 --device cuda
```

### RoadTrack live stop comparison

RoadTrack tracks vehicles on the projected road surface and estimates ground speed with a Kalman filter. I ran the original event detector and evaluator at the authors' published revision. The checkout contains 201 positive and 800 negative sequence entries across six fixed cameras, with 311 labeled vehicles in the positive clips. Six negative detection files are empty. The remaining negatives provide 19.85 camera-hours of replay. Both pipelines used the released YOLO11m detections at 10 FPS and the authors' exclusion rectangles. RoadTrack also used its camera homographies. The project replay used ByteTrack with a 0.30 activation threshold and the existing 3 s, 4 px stop rule. Inference from raw video is outside this comparison. [Paper](https://doi.org/10.1007/s11760-026-05148-1) · [Author release](https://github.com/corfyi/RoadTrack/tree/1a72b2613cf4e06725ba25d72ca7aaddb6338772)

The shared alert score matches each first confirmation to one labeled vehicle if the boxes overlap by at least 0.5 IoU on that frame. An alert before the labeled interval, outside the interval, or on a different vehicle is unmatched. A RoadTrack alert track is counted again after a gap over 2 s. Delays are relative to the first labeled frame and summarized over the matches for each method.

| Detector | Matches | Unmatched alerts | Missed labels | Precision | Recall | F1 | Negative alerts / hour | Median matched delay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack + pixel stop rule | 229 | 729 | 82 | 23.9% | 73.6% | 0.361 | 2.12 | 6.3 s |
| RoadTrack | 190 | 296 | 121 | 39.1% | 61.1% | 0.477 | 0.30 | 4.7 s |

The 24-sequence pilot used the first two positives and negatives per camera. Its F1 was 0.750 for RoadTrack and 0.537 for the pixel rule. On the full release, RoadTrack produced fewer unmatched alerts on every camera. K287 and K295 account for 40 fewer matched vehicles combined. In 119 of RoadTrack's 121 misses under the shared rule, an alerting track fired before the labeled interval and later overlapped the labeled vehicle by at least 0.5 IoU. The other two had no later overlap. The 10 km/h speed cutoff can trigger before a vehicle comes to a full stop. The authors' own trajectory-coverage evaluator produced 0.956 F1 and 2.91 s mean latency for RoadTrack. Those values use their matching rules and are kept separately from the first-confirmation scores. [Run record](runs/roadtrack.json)

RoadTrack is the baseline for improving precision in the live stop path. The next alert evaluation should measure how early a warning can arrive and still be useful, using the same time window for both methods. The six-camera replay reuses the release's detections and calibration. A raw-video evaluation on held-out cameras should confirm whether those gains carry over.

```sh
git clone https://github.com/corfyi/RoadTrack.git build/upstream-roadtrack
git -C build/upstream-roadtrack checkout 1a72b2613cf4e06725ba25d72ca7aaddb6338772
python -m pip install -r requirements-trackers.txt filterpy==1.4.5 lap==0.5.13 supervision==0.32.3 scipy==1.18.1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.road_validation.roadtrack build/upstream-roadtrack --per-camera 0 --jobs 8
```

### Whole-system reference for the three event types

The WIUT 2026 fixed-road-camera qualifier includes timed red-light running, wrong-way driving, and stopped vehicles. Team404 publishes code for all three. The organizer posts the overall hackathon winners, and I could not verify a public class-by-class qualifier ranking. BlockVision dropped wrong-way from its submitted class list. Its released predictions reproduce a 0.122 development score against its own labels with the organizer's evaluator. The AI City 2026 FETV winner, UniTraffic-Agent, scored 0.4848 on a composite for short fisheye clips that includes red-light and wrong-way classes. FETV has no stopped-vehicle class. [WIUT qualifier requirements](https://github.com/Pac1FF1zM/traffic_checker/blob/main/HACKATHON_REQUIREMENTS.md) · [WIUT results](https://hackathon.wiut.uz/results/) · [BlockVision results](https://github.com/SayyeedDeveloper/TXVU-Hackathon-BlockVision-team#results-on-the-sample-videos) · [FETV protocol](https://www.aicitychallenge.org/2026-track3/) · [AI City results](https://www.aicitychallenge.org/2026-challenge-winners/)

I ran Team404's raw-video pipeline with its own YOLO11n, tracker, and event rules on the same four development clips as our RF-DETR-S and ByteTrack pipeline. Both received the established lane and stop-line geometry and processed every decoded frame. The external run used its visual signal classifier and a manually drawn signal crop. Our pipeline used the clip's labeled signal phase. Team404 read red in 17 of the 33 frames. The accident model was disabled for this comparison. [Team404 source](https://github.com/Pac1FF1zM/traffic_checker/tree/d12fe4ed33f48906b6dae1a086a56a97529561b3) · [Clips and labels](road_validation/README.md)

| Clip | Reference onset | Team404 matches / extra alerts | Project matches / extra alerts |
| --- | ---: | ---: | ---: |
| Foggy wrong way | 2.0 s | 0 / 0 | 1 / 0 |
| Shadowed wrong way | 3.0 s | 0 / 0 | 0 / 0 |
| Highway stop | 5.0 s | 0 / 0 | 1 / 1 |
| Red-light crossing | 1.8 s | 0 / 1 | 1 / 0 |

The score checks only the target class per clip and matches onset within ±1 s for wrong way and stopped vehicles and ±0.5 s for red light. The labels and Team404 outputs in this check cannot establish vehicle identity. These four clips support a development comparison with a system that covers the same events. They cannot rank systems across new cameras. [Run record](runs/wiut-team404.json)

```sh
git clone https://github.com/Pac1FF1zM/traffic_checker.git build/upstream-wiut-team404
git -C build/upstream-wiut-team404 checkout d12fe4ed33f48906b6dae1a086a56a97529561b3
python build/upstream-wiut-team404/scripts/download_weights.py
python -m pip install -r requirements-trackers.txt -r requirements-rfdetr.txt ultralytics==8.3.236
python -m benchmarks.road_validation.wiut build/upstream-wiut-team404 build/road build/assets/rf-detr-small.pth --device cuda
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
