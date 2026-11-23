# Wildlife detection serving comparison

A Flask application for uploading one image, running it against two TensorFlow Serving detectors, and inspecting the resulting bounding boxes. The request invokes SSD and Faster R-CNN sequentially. The UI and REST endpoints expose individual model results and recorded elapsed times.

## What the repository can reproduce

```bash
python benchmark_logs.py
```

This command summarises the bundled CSV without running either model. The recorded sample contains 41 requests and 13 distinct original filenames.

| Logged elapsed time | SSD | Faster R-CNN |
|---|---:|---:|
| Median | 2.624 s | 2.777 s |
| 95th percentile | 14.595 s | 6.952 s |
| Requests where it was faster | 23 | 18 |

The median within-request difference (Faster R-CNN minus SSD) is 0.056 seconds. The timings cover image preparation, HTTP inference and result processing. They include an unknown mix of warm and cold requests. The CSV has no per-model success flag or hardware metadata, and filenames repeat. These results characterise the historical log, with no claim about model-only speed, accuracy or behaviour on new data. [Recorded counts and input hash](benchmark_results.json).

## Run inference with model weights

The Git checkout includes SavedModel graphs and label maps. It requires matching TensorFlow variable files to serve predictions. Install both complete SavedModels under `models/ssd/1/` and `models/faster_rcnn/1/`, including their `variables/` directories. Keep their output signatures and class IDs consistent with the app.

The inference path requires one TensorFlow Serving prediction object with equally sized `detection_boxes`, `detection_scores` and `detection_classes` arrays. Active detections require finite, ordered box coordinates, probability scores and integer class IDs. Optional `num_detections` limits processing to the active entries, excluding padding. A valid response with zero detections succeeds. Missing fields, inconsistent array lengths and invalid values return a failed inference and the upload route displays a model failure warning.

Run the response and upload-route regression tests with `python -m unittest test_inference_response -v` in an environment with the Flask application dependencies. These tests use generated fixture images and mocked model responses. Live validation with the complete trained weights remains separate. The legacy CSV continues to contain timings without model success flags.

```bash
docker compose up --build
curl http://localhost/health
curl -F file=@wildlife.jpg http://localhost/inference
```

The [original Docker image listing](https://hub.docker.com/r/ahakrami98/object-detection-comparison) is retained as a legacy distribution reference. The offline analysis requires Python and the checked-in CSV.

[API](docs/API.md) · [Architecture](docs/ARCHITECTURE.md) · [Training notebook repository](https://github.com/amirosseinz/sub-saharan-wildlife-object-detection)

The serving label map uses lion for class 3. The companion training pre-processing notebook lists eland. Model lineage and evaluation inputs must be reconciled before associating the published training AP with the serving timings.
