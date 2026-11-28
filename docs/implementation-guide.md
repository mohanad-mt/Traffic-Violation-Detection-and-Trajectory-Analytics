# Rebuilding the traffic pipeline

Start with the track-only fixture. Work backward from its alert, then replace the known observations with the output of a detector and a tracker. This order makes it possible to identify which stage caused a miss.

## From pixels to boxes

The core interface is `BoxDetector.predict_boxes` in `traffic_trajectory/detection.py`. Both neural adapters return the same box fields in full-frame pixels. A crop has its own coordinate origin, so the wrapper shifts every box back before merging. A detector can run once over the whole frame or also run over overlapping crops.

Read `yolox.py` with a 640-pixel input in mind. The model predicts offsets and sizes at a grid of positions at strides 8, 16 and 32. For one grid cell with index `(i, j)` and stride `s`, the predicted center is `s * (i + dx, j + dy)` and the width is `s * exp(dw)`. The height uses the corresponding height offset. Divide these values by the input resize scale to recover source pixels. The code clips boxes at the frame boundaries.

Objectness estimates whether a proposal contains an object. The classifier estimates the class for that proposal, and the YOLOX adapter multiplies the two output scores. Non-maximum suppression (NMS) starts with high-score boxes and removes overlapping lower-score ones. Intersection over union (IoU) is intersection area divided by union area: two disjoint boxes score 0 and equal boxes score 1. The adapter suppresses overlapping boxes once per view and again after merging tiles. The RT-DETR model instead learns to produce a ranked set of object queries, and its standard processor decodes the coordinates and scores.

To reproduce the main trade-off, compare full-frame and 640-pixel crops at the same full-frame cutoff. The crop lets distant objects occupy more inputs to the network. It also removes context and can create duplicates or fragments. Raising the crop cutoff changes the balance. Check the same experiment on a second corpus to see whether the effect depends on the scene.

## From detections to tracks

The point tracker predicts a bottom center from the last two measured positions and their time difference. It greedily pairs that prediction with the closest detection within a gate. This can consume the only plausible box for a different vehicle. A global association algorithm instead minimizes the total assignment cost for the group. Hungarian assignment gives one match per row and column. A gate or explicit unmatched option prevents forced implausible pairs.

In a Kalman filter, the prediction step combines the previous state with a motion model, often position plus velocity. It propagates the covariance through the same motion model and adds process noise. On a matched measurement, the update corrects the prediction by the weighted residual between the observed and predicted box. A more uncertain prediction gives the box more influence. Over a gap the covariance grows because the tracker has less direct evidence.

ByteTrack keeps high-score and low-score detections in separate association passes. A low score can extend an existing track when its geometry is consistent. OC-SORT uses observations to stabilize direction and correct an extrapolated track after an occlusion. In this repository, the library tracker assigns IDs and the adapter retains the detector's original box and confidence. An observation may be absent even while the internal track survives. Watch for that distinction when debugging a missed alert.

Read `tracker.py` first and implement its two-stage point association. Then replace its update with the library adapter in `box_tracking.py` and replay the same boxes. Measure HOTA for detection and association together, DetA and AssA for their components, IDF1 for identity continuity, and ID switches and fragments to diagnose different ways a trajectory breaks. The GRAM replay assembles that input for TrackEval.

## Camera geometry and event rules

A road direction is a vector, and the signed dot product of a movement vector with the legal direction measures progress along it. The cosine test compares headings independently of magnitude. The wrong-way rule also requires a minimum duration and projected displacement. Multiplying a pixels-per-second threshold by the time between observations makes the step check consistent across sampling rates.

The stationary rule keeps an anchor fixed during one stop episode. A timeout, a lane exit, or motion outside the radius can end the episode. For a red crossing, compute the signed distances of two consecutive points from the line. If the sign changes, linear interpolation estimates when the vehicle crossed. A finite line, the expected approach direction, the margin and the synchronized signal interval each reject a different false alert.

A homography maps a point on one plane to another by multiplying the homogeneous vector `[x, y, 1]` by a 3 × 3 matrix and dividing the first two coordinates by the third. Four non-collinear point pairs are the minimum. Additional measured pairs help fitting and independent check points reveal error. Close to the horizon or away from the measured region, division by a small third coordinate can amplify box jitter. Exported velocity is the difference between projected positions over the elapsed time. Treat the unit and calibration error as part of the measurement. A simple smoother can reduce jitter, at the cost of lag near red phase changes.

## Evaluating the full pipeline

Begin with the numerator and denominator. Box precision divides correct boxes by predicted boxes. Box recall divides correct boxes by labeled objects. Average precision summarizes the ranked precision-recall curve at a fixed IoU. COCO AP averages across 0.50 to 0.95 IoU and can be broken down by object size. Class-agnostic AP answers whether there was a vehicle at the right location. Class-aware AP also asks whether it had the right category.

An alert match has a different definition. The onset must be close to the label, and the predicted trajectory must agree with the reference vehicle when identity evidence is available. One-to-one assignment credits each label at most once. Count extra alerts per covered video hour, including negative footage, and measure both onset error and confirmation delay explicitly. Keep the onset tolerance and spatial radius fixed before running the test cameras.

Cache detector boxes at a low cutoff and replay tracker and event settings against them. A detector score is a ranking signal. To interpret it as a probability, group predictions by score on separate annotated data and compare each group to the fraction that actually matches. Tune for a useful operating point such as maximum recall subject to an observed precision floor. Split by camera or sequence to keep adjacent frames together. The highest validation F1 is a decision to test, and the independent test is the evidence for generalization.

## GPU and software checks

The ONNX Runtime path and OpenCV path consume the same tensor and share preprocessing, decode and filtering. Warm up both runs and compare raw outputs before comparing tracks and alerts. Returning outputs to CPU synchronizes the CUDA work in these benchmarks. End-to-end time includes decoding, preprocessing, model calls, filtering, tracking and event rules. Extra tiling or CPU-to-GPU copies can erase an inference-only speed gain.

Check these questions while rebuilding the project.

1. Where should a stop-line crossing time change if you alter the detection rate? Which signals stay in seconds?
2. Can a higher AP detector yield fewer matched events? Trace the fog target through its low-score boxes, track births and wrong-way state.
3. Why did ByteTrack improve identity scores on GRAM while the best alert setting used OC-SORT? Compare the benchmark inputs and error definitions.
4. How can two predictions with identical event type and onset still have different evaluation outcomes? Add a second vehicle in the same lane and run identity-required scoring.
5. Which arrays stay in memory in streaming mode? Follow their expiration at a tracking gap. Then inspect what full visualizations still require offline.

Run the [upgrade benchmarks](../benchmarks/UPGRADE.md) yourself and retain one prediction file and one failed case for each layer.
