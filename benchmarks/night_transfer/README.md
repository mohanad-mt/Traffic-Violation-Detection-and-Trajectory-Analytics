# Nighttime transfer check

Two separate camera views from the NJIT demo were reserved for a frozen test after choosing the detector and tracker settings on the development clips. The onset labels come from the source log, which is rounded to seconds. The four reference points per clip follow the approximate bottom center of the vehicle indicated by the source overlay. Camera geometry was marked on the clips before running either pipeline. Both use a 1 s onset tolerance and a 12 px radius.

| Case | Source | Extract | Source SHA-256 |
| --- | --- | --- | --- |
| Camera 1 | [South-Korea-Wrong-Way-1.mp4](https://web.njit.edu/~cliu/NJDOT/wrong-way/South-Korea-Wrong-Way-1.mp4) | 0–8 s | `b4baa6053d4ce12c7d36f997125d1067089c8903c11eb8a977b43193c7f57800` |
| Camera 2 | [South-Korea-Wrong-Way-2.mp4](https://web.njit.edu/~cliu/NJDOT/wrong-way/South-Korea-Wrong-Way-2.mp4) | 15–23 s | `f42602e3a49ac12480a1a0e9034e260a5d291e695e28f27b6d5f777515c78e8e` |

Both pipelines missed both reference events and produced zero alerts. The footage is low resolution, the target is partly represented by headlight glare, and the original overlays remain visible. [Source timestamps](https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong-way.txt)

```sh
python benchmarks/wrong_way/prepare_clip.py South-Korea-Wrong-Way-1.mp4 build/camera-1.avi --start-frame 0 --source-frames 120
python benchmarks/wrong_way/prepare_clip.py South-Korea-Wrong-Way-2.mp4 build/camera-2.avi --start-frame 225 --source-frames 120

for camera in camera-1 camera-2; do
  python -m traffic_trajectory "benchmarks/night_transfer/$camera.scene.json" --video "build/$camera.avi" --model object_detection_yolox_2022nov.onnx --device cuda --include-tracks > "build/$camera.baseline.json"
  python -m traffic_trajectory "benchmarks/night_transfer/$camera.scene.json" --video "build/$camera.avi" --model build/assets/rtdetr-r18 --detector rtdetr --tracker ocsort --confidence 0.20 --device cuda --include-tracks > "build/$camera.candidate.json"
  for run in baseline candidate; do
    python -m traffic_trajectory.evaluate "build/$camera.$run.json" "benchmarks/night_transfer/$camera.labels.json" --identity-required --radius-px 12 --tolerance-s 1
  done
done
```
