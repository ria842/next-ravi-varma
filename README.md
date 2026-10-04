# The Next Ravi Varma

A text-to-image generative pipeline that synthesises novel paintings in the style of
**Raja Ravi Varma** (1848–1906) from natural-language descriptions of scenes from the
Indian epics.

Inspired by *The Next Rembrandt* (2016), but replacing that project's rigid structural
averaging with modern latent diffusion and parameter-efficient fine-tuning. You describe a
scene from the Mahabharata or the Ramayana; the system returns a high-fidelity digital
painting carrying Varma's oil textures, palette, soft skin rendering and theatrical
framing.

> Academic project — BITS, Semester 7, Artificial Intelligence.

## What it does

```
"Yudhishthira weeping over the fallen body of Karna on the
 battlefield of Kurukshetra, dramatic shadows"
                        |
                        v
         LLM prompt expansion (+ style trigger token)
                        |
                        v
    SD 1.5 + Ravi Varma LoRA  <---- ControlNet (OpenPose / Canny)
                        |                      ^
                        v                      |
              generated painting        pose reference
```

## Architecture

| Component | Choice | Why |
|---|---|---|
| Base model | Stable Diffusion 1.5 (photoreal checkpoint) | Trains in ~40 min on a free T4; mature ControlNet ecosystem |
| Style injection | LoRA, rank 32, UNet only | Parameter-efficient; no catastrophic forgetting |
| Captioning | BLIP-2 + human correction pass | Content-only captions so style binds to the trigger token |
| Prompt expansion | Hosted LLM API + deterministic fallback | No GPU headroom for a local 7B beside the diffusion pipeline |
| Structural control | ControlNet OpenPose / Canny | Enforces Varma's balanced, static, dignified poses |
| Interface | Gradio (`share=True`) | Tunnels out of Colab cleanly |

SDXL was evaluated and rejected: 6–10× slower on a free T4, OOM-prone at 1024 px with no
bf16 support, and a less mature ControlNet family. It remains listed as future work.

## Milestones

| # | Milestone | Deliverable | Status |
|---|---|---|---|
| 1 | Corpus curation & multimodal captioning | Paired image/caption dataset + data card | **Done** |
| 2 | Style-injected LoRA fine-tuning | LoRA weights + loss curves | In progress |
| 3 | Prompt parsing & semantic scene generation | Prompt expansion module + generation script | Not started |
| 4 | ControlNet guardrailing & evaluation | Gradio dashboard + evaluation report | Not started |

## Dataset

Sourced from Wikimedia Commons (`Category:Paintings by Raja Ravi Varma`, ~168 files).
Ravi Varma died in 1906, so the original works are in the public domain.

Two deliberate curation decisions:

- **Ravi Varma Press oleographs are excluded.** The chromolithographs are plentiful online
  but look flat and printed, and blur the oil-painting aesthetic the model is meant to
  learn.
- **~20 paintings are held out** from training, stratified across subcategories, to
  support style-similarity evaluation.

Expected working set after dedup and filtering: **~110–130 images**.

The dataset is not committed to this repository — see `scripts/` for the scraper that
rebuilds it.

## Evaluation

- **CLIP-T** — alignment between the generated image and the input prompt
- **Style similarity** — CLIP/DINOv2 embedding distance to the held-out Varma paintings
- **Human rubric** — 3 raters × 20 images, scoring style fidelity and prompt fidelity
  separately

FID is deliberately **not** reported: at n ≈ 130 the estimator bias dominates and the
resulting number carries no signal.

## Setup

> Training requires a GPU. All training runs happen on Kaggle (12 h sessions, 30 h/week)
> or Google Colab. Local machines are used for data preparation, caption review and code.

```bash
git clone https://github.com/ria842/next-ravi-varma.git
cd next-ravi-varma
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Build the dataset (Milestone 1)

```bash
python scripts/scrape_commons.py --dry-run   # metadata + classification, no downloads
python scripts/scrape_commons.py             # full pipeline, ~10 min
```

Discovers files via deep-category search, separates oils from Ravi Varma Press prints,
drops crops and low-resolution scans, downloads width-capped renderings, removes
near-duplicates by perceptual hash, and writes a stratified train/holdout split.
Re-running is safe — completed downloads are skipped.

Outputs `data/curated/oil/`, `data/manifest.jsonl` and `data/DATA_CARD.md`. None of it
is committed; rebuild it from the script.

Useful flags: `--min-edge` (short-edge floor, default 460 — raise to 700 only if you
move to 768px training) and `--max-width` (download cap, default 2048).

### Caption the corpus (Milestone 1)

```bash
python scripts/caption_corpus.py draft --limit 10   # GPU: smoke test on 10 images
python scripts/caption_corpus.py draft              # GPU: all 120
python scripts/caption_corpus.py import             # or adopt captions made elsewhere
python scripts/caption_corpus.py review --shard 1/3 # local: one teammate's share
python scripts/caption_corpus.py lint
python scripts/caption_corpus.py export
```

`import` adopts a `file_name`/`text` jsonl produced by another tool, applies the
hand corrections in `data/caption_overrides.jsonl` on top, and injects the
ground-truth entity clause (`Shakuntala`, `a scene from the Mahabharata`) that
no VLM can supply — falling back to the bare names when the full clause would
breach the token budget, and skipping rather than truncating. Everything it
adopts is marked unreviewed, because §1.3 discipline is the human pass.

`draft` needs a GPU — run it in Colab or Kaggle. BLIP-2 answers a fixed
three-question template per image (figures and clothing, pose and gaze,
background), and the answers are assembled into one caption behind the trigger
token `rvrma`. Named figures come from the file title and Commons categories,
not from the VLM: BLIP-2 will never guess "Shakuntala".

Install `transformers` locally for exact CLIP token counts; without it the
budget check falls back to a calibrated estimate that errs about 7 tokens high,
which can cost you an entity clause that would in fact have fit.

`import`, `review`, `lint` and `export` run locally with no GPU. `review` is resumable and
shardable, so three people can split the correction pass without collisions.

**Captions describe content, never style.** A caption containing "oil painting"
or "soft lighting" binds Varma's style to those common English tokens instead of
to the trigger, and the trigger then does nothing at inference. This is enforced,
not just documented: `review` refuses to save a caption that trips the style
lint, and `export` refuses to emit a corpus that is incomplete, unreviewed or
failing lint.

Captions live in `data/captions.jsonl`, deliberately separate from
`manifest.jsonl` — the scraper rewrites the manifest on every run and would
otherwise discard the human correction pass. `export` writes `data/train/` and
`data/holdout/`, each with the `metadata.jsonl` that `diffusers`' `imagefolder`
loader expects.

### Train the LoRA (Milestone 2)

Needs a GPU. Open `notebooks/milestone2_lora_colab.ipynb` in Colab on a T4 —
it clones this repo, mounts Drive and calls the trainer. The training logic
lives in the script, not the notebook, so a run is reviewable and reproducible.

```bash
python scripts/train_lora.py --output-dir runs/r32          # the reference run
python scripts/train_lora.py --rank 16 --output-dir runs/r16        # ablation
python scripts/train_lora.py --no-bucket --output-dir runs/square   # ablation
python scripts/train_lora.py --output-dir runs/r32 --resume latest  # after a kill
```

Defaults are the hyperparameters in the plan: rank 32, alpha 32, lr 1e-4
constant-with-warmup, effective batch 4 by gradient accumulation, fp16, 8-bit
Adam, gradient checkpointing, **text encoder frozen**, 750 steps.

**Steps are not epochs.** Each step consumes `--gradient-accumulation-steps`
images, so at the default accum 4 over 100 images an epoch is 25 steps. The
plan's 2500 assumed batch 1 with no accumulation; run at accum 4 it is 100
epochs, and run 1 collapsed under it — validation samples were clean at step 500
and near-empty smears by 2500, with training loss bottoming near step 1200 and
climbing after. The default is now 750 steps (30 epochs); the script prints the
epoch count and warns past 40.

**Latents and text embeddings are cached once**, after which the VAE and text
encoder move to CPU. Only the UNet and the adapters stay on the GPU, which is
what makes rank 32 fit in a free T4's 16 GB with headroom.

**Resolution is 512, not 768.** 49 of the 100 images have a short edge below 512
and the median is 528 — training at 768 would upscale most of the corpus and
teach the LoRA interpolation artefacts as though they were brushwork.

**Aspect-ratio bucketing is on by default.** 93 of 100 paintings are portrait,
median ratio 0.70, so a square centre crop keeps a median 70% of each painting
(worst case 54%); bucketing keeps 92% and upscales nothing. It also keeps the
captions honest — they carry a *composition* field, and a crop would describe a
framing the model never sees. `--no-bucket` restores square cropping for the
baseline comparison.

**Checkpoints every 125 steps**, each carrying the optimiser moments as well as
the adapter weights, so a resume continues the schedule instead of restarting
it. `--checkpoints-total-limit` prunes only `training_state.pt`; **adapter
weights are never deleted**. Run 1 lost its best model to a keep-the-last-3
policy, which is precisely the wrong thing to prune on a run that may
overtrain. Point `--output-dir` at Drive: Colab kills sessions, and `loss.csv` is
flushed every logging step for the same reason. The loss curves are as much the
Milestone 2 deliverable as the weights.

Notebook sections 6 and 7 answer the top entry in the plan's risk register —
whether the `rvrma` trigger does anything — by generating the same seed and
scene three ways (no LoRA / LoRA without the trigger / LoRA with it) and then
sweeping the adapter scale. Run that on the **first** checkpoint, not at the end
of the week.

## Repository layout

```
.
|- README.md
|- scripts/
|   |- scrape_commons.py   # corpus discovery, curation, dedup, split
|   |- caption_corpus.py   # BLIP-2 drafting, human review, lint, export
|   |- train_lora.py       # LoRA training: latent cache, AR buckets, checkpoints
|- src/            # prompt expansion, generation, ControlNet pipeline
|- notebooks/
|   |- milestone2_lora_colab.ipynb   # T4 driver: Drive, dataset, run, curves
|   |- requirements-gpu.txt          # GPU deps; torch deliberately unpinned
|- eval/           # metric harness and rating study
|- app/            # Gradio dashboard
```

Data, model weights and internal planning documents are untracked; see `.gitignore`.

## Licence & attribution

Raja Ravi Varma's paintings are in the public domain. Generated outputs are derivative
stylistic works produced for academic study, and are not presented as, nor to be
represented as, authentic works by the artist.
