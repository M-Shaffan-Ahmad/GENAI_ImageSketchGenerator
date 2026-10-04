# Ready-to-read six-minute video script

Presenter: Muhammad Shaffan Ahmad, 23i-0673, Section A, FAST NUCES.
Target length: 5–7 minutes. Read the quoted narration naturally and pause for the
on-screen actions; timestamps are approximate. Use one clean flower photo for the
four tasks so the comparisons are easy to follow. Show the real model outputs.

## Before recording

Have the browser, a clean flower photograph, public GitHub repository,
`report/main.pdf`, and the MLflow viewer ready. Use a readable browser zoom.
Start MLflow in a separate terminal before recording:

```bash
venv/bin/python scripts/relocate_tracking.py
venv/bin/mlflow ui --backend-store-uri sqlite:///experiments/tracking.db --host 127.0.0.1 --port 5000
```

Open http://localhost:5000. Open the app at http://localhost:18081. Keep the paper's
Optuna/training figures and final test tables easy to reach.

## 0:00–0:45 — Introduction and Docker startup

**Show:** Your public repository, then a terminal. Run:

```bash
FRONTEND_PORT=18081 BACKEND_PORT=18082 docker compose up --build
```

Use another terminal or switch to the browser once the containers are ready. These
images were already built, so do not describe the cached startup as a first build.

**Say:**

> Assalamualaikum. My name is Muhammad Shaffan Ahmad, student ID 23i-0673,
> Section A, from FAST NUCES. This is my Generative AI assignment: an image
> restoration and sketch generation studio.
>
> The project has four workspaces: universal restoration, hard-routed restoration,
> a soft mixture of experts, and object-to-sketch generation. The models were
> trained in PyTorch using Google Colab, with Optuna for hyperparameter search
> and MLflow for experiment records.
>
> Here I am starting the application with Docker Compose. These images have
> already been built. The React frontend communicates with a FastAPI backend,
> which runs the exported ONNX models on the CPU.

## 0:45–1:40 — Task 1: Universal restoration

**Show:** Universal Restoration. Upload your clean flower photograph. Select
salt-and-pepper noise, set probability to 0.08, click **Restore image**, then point
to input/output/reference/error panels. Download the PNG and open it.

**Say:**

> First, I upload a clean flower photograph and simulate salt-and-pepper noise.
> This gives us a known reference for checking the reconstruction. For an
> already-damaged upload without a clean reference, the application cannot know
> the true missing details.
>
> Task one uses a single convolutional autoencoder trained for noise, Gaussian
> blur, and rectangular occlusion. Its loss combines L1 pixel error with an SSIM
> structural term. L1 penalizes differences in pixel values, while the SSIM term
> encourages similar local structure.
>
> These panels show the corrupted input, restored output, clean reference, and
> absolute error map. I can also download the result as a PNG. The models operate
> at 128 by 128 pixels, so a larger preview does not create extra detail.

## 1:40–2:25 — Task 2: Hard routing

**Show:** Hard-Routed Restoration. Keep the same photograph and noise setting;
restore it and point to the four probability bars. Then choose **Use image as
uploaded** and restore again. Describe the route actually displayed.

**Say:**

> Task two first classifies the actual input as clean, noisy, blurred, or occluded.
> It then selects one of three separately trained specialist restoration models.
>
> These bars show the classifier's probabilities. They are predictions from the
> input image, not simply a copy of the corruption option I selected.
>
> Now I remove the simulated corruption. When the classifier predicts clean,
> the identity branch returns the input unchanged. This helps preserve images
> that do not need restoration. However, a wrong prediction can send an image
> to the wrong specialist, so routing accuracy affects the final result.

## 2:25–3:15 — Task 3: Soft mixture of experts

**Show:** Soft Mixture-of-Experts. Select Gaussian blur, kernel 5, sigma 1.5.
Restore and point to the four weights and images. If time allows, increase sigma
to 2.5 and compare the new weights. Explain the values actually shown.

**Say:**

> Task three replaces a single hard choice with a learned weighted combination.
> It blends the identity branch with the noise, blur, and occlusion specialists.
> The four contribution weights sum to one.
>
> Here I apply Gaussian blur and inspect the contribution of the blur expert
> and the unchanged input. The displayed weights tell us how this particular
> result was produced.
>
> Training first warms up the gating network, then jointly fine-tunes the gate
> and experts. In the final tests, soft blending improved average PSNR and SSIM
> over hard routing across the corrupted severity groups. Mild blur is still
> a limitation: processing can remove useful detail and make an image worse
> than leaving it unchanged.

## 3:15–4:10 — Task 4: Sketch generation

**Show:** Object-to-Sketch Generator. Keep the same photograph. Run **Fine Pencil**,
**Technical Ink**, and **Tonal Charcoal**, allowing each result to finish. Download
one sketch.

**Say:**

> Task four generates a sketch from the photograph. I will run the same input
> with Fine Pencil, Technical Ink, and Tonal Charcoal so we can compare styles.
>
> This task uses a conditional GAN with a U-Net generator and a style-conditioned
> discriminator during training. A categorical style embedding controls the
> requested drawing style. The generator combines an adversarial loss with
> paired L1 error.
>
> The targets are generated by image-processing algorithms, rather than drawn
> by human artists. The evaluation therefore measures agreement with those
> synthetic targets. The three styles produce distinct outputs, but technical
> ink has the weakest structural similarity in the final tests. At deployment,
> only the generator is needed to produce the sketch.

## 4:10–5:10 — Training, search and final evaluation

**Show:** MLflow parameters, a training/validation curve and artifacts from an
original Colab evidence import. Show the paper's Optuna plot and final test tables.
If available, point to the classifier confusion matrix and gate heatmap.

**Say:**

> This is the MLflow viewer. These training records are explicitly labeled as
> retrospective imports of the original Colab evidence. The original Optuna
> studies are also preserved, so the search results and selected settings can
> be inspected.
>
> Hyperparameters and checkpoints were selected using validation data. The
> selected models were then frozen for the final test comparison, with 61,490
> restoration views and 1,068 sketch pairs.
>
> The classifier achieved approximately 97.36 percent test accuracy and a macro
> F1 score of 0.959. Restoration is evaluated using PSNR, SSIM, and L1 error,
> with results separated by corruption and severity. Sketch results are reported
> separately for each style. This makes the weaker cases visible instead of
> hiding them behind one overall score.

## 5:10–6:00 — Design, limitations and conclusion

**Show:** The supplied Stitch screen in the paper beside the working app, a mild-blur
or ink failure figure, then the public repository and model download link.

**Say:**

> The interface was revised using the supplied Google Stitch export. The paper
> distinguishes the original design from the working application. The app shows
> actual CPU inference, real model outputs, and measured latency.
>
> The project is functional, but restoration is not perfect. Mild blur can lose
> detail, occluded regions cannot always be recovered correctly, and some
> generated sketches contain grid artifacts or inaccurate ink contours.
>
> The IEEE report contains the methods, search results, training curves, final
> test tables, failure analysis, and an AI-assistance appendix. Source code,
> exported models, and selected PyTorch checkpoints are available in this public
> GitHub repository. Thank you.

## After recording

Upload the recording to YouTube as **Unlisted**, test the link signed out, and put
it into `report/submission_metadata.tex`. Update the paper's sentence saying the
personal demonstration is pending, recompile, and submit the final PDF through
Classroom. Follow `docs/submission_steps.md` for those final steps.
