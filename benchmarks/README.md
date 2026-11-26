# Experiment log

Keep each run reproducible. Record the code commit, clip and model checksums, event labels, sampling rate, lane geometry, detector and tracker settings, matching tolerance, and measured results. Record why a setting changed and whether you chose it before or after seeing the clip's results.

## Recorded runs

| Run | What changed | Result | Interpretation |
| --- | --- | --- | --- |
| Synthetic integration | 10 FPS, clean background reference, configured red interval, default event thresholds | 3 true positives, 0 false positives, 0 false negatives at ±0.5 s | Exercises video decoding, tracking, all three rules, and scoring. The generated clip includes a crossing outside the red interval. |
| Nighttime road clip | Pretrained YOLOX-S at confidence 0.35, 10 FPS, one annotated lane | 0 labeled events and 0 alerts across 80 frames | Verifies real-image inference and a negative case. Recall is undefined. [Reproduce the run](nighttime/README.md). |
| NJIT wrong-way baseline | YOLOX-S at 0.35, 7.5 FPS, minimum step 0.6 px for the low-resolution scene | 0 of 2 reference events, 0 false positives at ±1.0 s | In this clip, the model produced too few usable observations for the wrong-way tracks. [Reproduce the run](wrong_way/README.md). |
| NJIT confidence diagnostic | Lowered confidence from 0.35 to 0.08 after inspecting detections | 1 of 2 reference events, 0 false positives, 1.60 s confirmation delay on the match | Detections increased from 159 to 332 across the full frame. The extra observations recovered one event. This was exploratory tuning on the same clip. |
| NJIT rule diagnostics | Reused the 0.08 tracks. Try a 0.5 s minimum duration, a wider lane polygon, then both | Each run still matched 1 of 2 with 0 false positives. Shorter duration reduced the matched delay to 1.47 s | The second candidate track remained too short or fragmented for the required reverse displacement. Keep the project defaults until a separate held-out set supports a change. |

The log covers a synthetic integration check, a real negative clip, and a small exploratory positive clip. For a real evaluation, freeze the settings on one camera set, then report scores on separate fixed-camera clips with labeled wrong-way, stopped-vehicle, and red-light events. Include the synchronized red intervals for each signalized clip.

The synthetic pipeline was introduced in GitHub commit `055a49c`. The nighttime and NJIT runs used the implementation in `f65117e`. The NJIT configurations leave the runtime defaults intact.
