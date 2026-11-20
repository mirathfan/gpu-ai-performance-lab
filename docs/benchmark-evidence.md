# Benchmark evidence

## Historical evidence

The committed ResNet18, YOLOv8n, and Monte Carlo reports are the available historical record. [Summary CSV exports](../reports/snapshots/README.md) retain their printed precision and cite the original commit and report hashes. They are not a replacement for raw measurements. Do not infer repeat-level variance, confidence intervals, or missing environment fields from these summaries.

## Publishing a measured run

1. Run the documented benchmark commands on the target hardware.
2. Copy the original benchmark output CSV/JSON files into `results/published/<run-id>/`. This nested path is not excluded by the top-level `results/*.csv` and `results/*.json` patterns. Keep model binaries and TensorRT caches out of Git.
3. Include a `README.md` recording the source commit (and whether the checkout was dirty), exact command, timestamp/timezone, GPU, OS, driver, CUDA, framework/provider versions, model/weights, workload shape, warmups, measured iterations, and repeat count. Mark missing fields as unknown.
4. Include correctness checks and disclose what timing includes: transfers, synchronization, compilation, preprocessing, and postprocessing.
5. Link the run from the main README and its report. Keep unsuccessful or slower comparisons visible. Never fill missing samples by reconstructing them from rounded summaries.
