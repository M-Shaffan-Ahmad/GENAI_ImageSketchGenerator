# GPU training in Google Colab

1. Open https://colab.research.google.com/ and choose **File → Upload notebook**.
2. Upload `task1_gpu.ipynb` from this folder.
3. Choose **Runtime → Change runtime type → GPU** and connect. GPU allocation depends on
   Colab availability; the notebook checks that CUDA is available before training.
4. Run the notebook cells in order. Authorize the Google Drive mount in your browser.
5. When asked for a source ZIP, upload `task1_source.zip` from this folder. Rebuild it
   after local source changes with `python package_colab.py`.
6. The notebook downloads Oxford data, rebuilds the same manifests, runs tests, tunes
   the spatial autoencoder on validation data, and trains the selected configuration.

The new default model stores a **32×16×16 feature map (8,192 values)** with no skips,
instead of a 512-value vector. Default encoder dropout is zero. Optuna may investigate
0–10% dropout and spatial latents of 4,096, 8,192 or 16,384 values. Training defaults to
60 epochs; actual image quality must be verified after GPU training.

The notebook copies checkpoints, studies and MLflow records to
`MyDrive/GenAI_Ass1/spatial_v2` after each completed epoch and at command completion.
SQLite databases are backed up through SQLite's backup API before copying them.
Recovery cells restore these files into a new runtime. Final training resumes from
`last.pt`, including optimizer and RNG states. Training images stay on runtime-local disk
to avoid thousands of Drive reads. An interruption during an epoch resumes from the
last saved epoch. If Drive backup fails, stop and resolve it before relying on persistence.

The notebook exports a new ONNX model and evaluates **validation**, not test, by default.
The test set has already been used for the first model's reported results. An explicitly
enabled final test cell records a further comparison; it is not a new unseen test set.

At the end, download `task1_spatial_results.zip`. Extract into this repository to obtain
`models/universal_spatial.onnx`, its metadata, training logs, study and validation results.
To use it in the app, set `MODEL_PATH=models/universal_spatial.onnx` and restart the backend.
The currently deployed `models/universal.onnx` remains the earlier vector model until a
new trained export is imported. The notebook does not connect this assistant to your
Google account; sign-in and runtime authorization happen through Colab.
