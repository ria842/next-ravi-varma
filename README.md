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
| 1 | Corpus curation & multimodal captioning | Paired image/caption dataset + data card | Not started |
| 2 | Style-injected LoRA fine-tuning | LoRA weights + loss curves | Not started |
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
git clone <repo-url>
cd next-ravi-varma
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
.
|- README.md
|- scripts/        # corpus scraper, dedup, caption pipeline
|- src/            # prompt expansion, generation, ControlNet pipeline
|- notebooks/      # Colab/Kaggle training notebooks
|- eval/           # metric harness and rating study
|- app/            # Gradio dashboard
```

Data, model weights and internal planning documents are untracked; see `.gitignore`.

## Licence & attribution

Raja Ravi Varma's paintings are in the public domain. Generated outputs are derivative
stylistic works produced for academic study, and are not presented as, nor to be
represented as, authentic works by the artist.
