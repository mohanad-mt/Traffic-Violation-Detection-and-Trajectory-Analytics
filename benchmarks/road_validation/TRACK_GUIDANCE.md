# Prior-frame guidance for vehicle detection

We tested whether a weak sighting in one frame could help RT-DETR-R18 recognize a vehicle in the next. CaTDet and SegTrackDetect both use a tracker to choose regions for a second detector pass. The prototype uses the previous proposal's center as a one-frame position estimate. It leaves motion prediction, longer occlusions, and retraining for separate experiments. [Research and implementation review](../RESEARCH.md#using-tracking-to-guide-detection)

Each frame used the same full-frame predictions. We then rechecked up to four crops centered on proposals above 0.06 in the configured lane. The test used fixed crop sides of 64 and 96 px. A crop box needed a score above 0.20, a bottom center within 16 px of its guide, compatible dimensions, and a bottom center inside the lane. The same-frame control used that frame's weak proposals and the same crop settings. No guidance or tracker accessed the hand trace during alert evaluation.

| Fog run | Nearby boxes at 7 reference times | Matched wrong-way alerts | Extra alerts, OC-SORT | Added detector time |
| --- | ---: | ---: | ---: | ---: |
| Full frame, 0.20 | 0 | 0/1 | 0 | — |
| Full frame, 0.03 | 4 | 0/1 | 5 | — |
| Same-frame crops, 64 / 96 px | 0 / 0 | 0/1 in both | 0 in both | 32.5 / 28.5 ms/frame |
| Prior-frame crops, 64 / 96 px | 0 / 0 | 0/1 in both | 0 in both | 32.6 / 27.9 ms/frame |

Nearby means within 12 px of the hand-traced bottom center. The crop runs used 0.20 for that count. The prior-frame crops added 5 and 9 above-threshold boxes across the 60-frame fog clip, with zero near the reference at the sampled times. Full-frame inference averaged 14.3 ms/frame on the RTX 5090. With the point tracker the 0.03 run produced eight extra alerts and the same zero matches. At 0.20, OC-SORT matched the red-light alert and issued zero alerts on the nighttime negative clip under all crop settings.

In a separate diagnostic we centered the crops directly on the hand-traced position at the seven times. The 96 px crop gave six nearby responses above 0.03. The best reached 0.132, and neither crop size produced a nearby response above 0.20. A known search location recovered more low-score responses, while this detector still struggled to classify the fog vehicle. The position-based diagnostic is excluded from the alert scores. These three clips are development cases. Keep the recheck as a benchmark and test changes to the detector or actual visual tracking separately.

## Reproduce

Set up RT-DETR and the optional trackers and prepare the existing road and night clips. [Setup and clip preparation](../UPGRADE.md#reproduce)

```sh
python -m benchmarks.road_validation.track_guided_recovery build/assets/rtdetr-r18 build/road build/nighttime/sample.avi --device cuda > build/track-guided-recovery.json
```

[Recorded scores and configuration](../runs/track-guided-recovery.json)
