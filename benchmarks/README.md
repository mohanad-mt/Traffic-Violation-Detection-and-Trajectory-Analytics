# Experiment log

| Run | Settings | Result |
| --- | --- | --- |
| Synthetic demo | 10 FPS, background subtraction, default event thresholds | 3 of 3 labels matched within ±0.5 s. No extra alerts, including for the vehicle that crossed after the red phase. |
| Nighttime road clip | YOLOX-S at 0.35, 10 FPS, one labeled lane | 0 labeled events and 0 alerts over 80 frames. [Clip and commands](nighttime/README.md) |
| NJIT wrong-way baseline | YOLOX-S at 0.35, 7.5 FPS, 0.6 px minimum step | 0 of 2 labels matched within ±1.0 s. No false positives. [Clip and commands](wrong_way/README.md) |
| NJIT lower confidence | Same scene, cutoff lowered to 0.08 after inspecting the baseline | 1 of 2 matched, with no false positives. Confirmation came 1.60 s after the labeled onset. |
| NJIT event rule trials | Reused the 0.08 tracks. Tested 0.5 s minimum duration, a wider lane polygon, and both together | All three trials still matched 1 of 2 with no false positives. Shorter duration reduced the matched delay to 1.47 s. |

Lowering the cutoff raised the observation count from 159 to 332 and recovered the first NJIT event. Detections of the second vehicle came in short fragments. Widening the polygon and shortening the duration did not recover it. The 0.08 cutoff was chosen on that clip, so its score is exploratory.

The demo runs the pipeline from GitHub commit `055a49c`. The nighttime and NJIT runs used `f65117e`. Each clip's notes include the data and model checksums, scene, labels, and commands.

Next evaluation needs held-out camera views with positive examples of all three events and synchronized red intervals for the signalized clips. AI City's 2021 anomaly track has timestamped stalls and crashes. [Dataset notes](https://www.aicitychallenge.org/2021-data-and-evaluation/#:~:text=Track%204%3A%20Traffic%20Anomaly%20Detection)
