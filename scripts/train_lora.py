#!/usr/bin/env python3
"""
Train the Ravi Varma style LoRA on Stable Diffusion 1.5.

Pipeline
--------
1. cache  : encode every training image to a VAE latent and every caption to a
            frozen CLIP embedding, once. The VAE and text encoder then move off
            the GPU for the rest of the run -- only the UNet and the LoRA
            adapters stay resident, which is what makes rank 32 at 512 fit on a
            free-tier T4.
2. train  : standard epsilon-prediction diffusion loss against the cached
            latents. Only the LoRA adapters receive gradients.
3. save   : a checkpoint every --checkpointing-steps, written to --output-dir
            (point that at Google Drive on Colab), plus loss.csv appended every
            logging step so the curve survives a killed session.

Aspect-ratio bucketing (plans/PLAN.md §2, Milestone 2)
------------------------------------------------------
93 of the 100 training paintings are portrait, median aspect ratio 0.70. The
upstream `diffusers` LoRA script resizes the short edge and then crops to a
square, which throws away ~30% of a typical Varma portrait -- and our captions
carry a composition field ("Centered figure", "Figure fills the frame") that a
crop quietly falsifies. So images are assigned to aspect-ratio buckets near the
target area instead, and the whole painting is trained on. --no-bucket restores
the square-crop behaviour for a baseline run.

Resolution
----------
Train at 512, not 768. 49 of the 100 images have a short edge under 512 and the
median is 528, so 768 would mean upscaling most of the corpus and learning
interpolation artefacts as style.

Usage
-----
    python scripts/train_lora.py --data-dir data/train --output-dir runs/r32
    python scripts/train_lora.py --rank 16 --output-dir runs/r16     # ablation
    python scripts/train_lora.py --no-bucket --output-dir runs/square
    python scripts/train_lora.py --output-dir runs/r32 --resume latest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRIGGER = "rvrma"
CLIP_MAX_TOKENS = 77

# UNet attention projections. The text encoder stays frozen for every run in
# plans/PLAN.md §3 Milestone 2 -- training it is what usually breaks the trigger.
LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


# ---------------------------------------------------------------------------
# aspect-ratio buckets
# ---------------------------------------------------------------------------

def make_buckets(resolution: int, step: int = 64, min_dim: int = 320,
                 max_dim: int = 1024) -> list[tuple[int, int]]:
    """Bucket sizes whose area stays at or under resolution**2, both orientations.

    Every side is a multiple of 64 because the VAE downsamples by 8 and the UNet
    by a further 8.
    """
    area = resolution * resolution
    out: set[tuple[int, int]] = set()
    for h in range(min_dim, max_dim + 1, step):
        w = (area // h) // step * step
        w = max(min_dim, min(max_dim, w))
        if w * h <= area:
            out.add((w, h))
            out.add((h, w))
    return sorted(out)


def pick_bucket(size: tuple[int, int], buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """The bucket closest in log aspect ratio, so error is symmetric in w/h."""
    w, h = size
    target = math.log(w / h)
    return min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - target))


def fit_to_bucket(img, bucket: tuple[int, int]):
    """Resize to cover the bucket, then centre crop off the small remainder."""
    from PIL import Image

    bw, bh = bucket
    w, h = img.size
    scale = max(bw / w, bh / h)
    nw, nh = max(bw, round(w * scale)), max(bh, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - bw) // 2, (nh - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    file_name: str
    text: str
    bucket: tuple[int, int]
    upscaled: bool
    latent_mean: object = None   # [4, h/8, w/8] fp16, CPU
    latent_std: object = None
    embedding: object = None     # [77, 768] fp16, CPU
    latent_mean_flip: object = None
    latent_std_flip: object = None


def load_metadata(data_dir: Path) -> list[dict]:
    path = data_dir / "metadata.jsonl"
    if not path.exists():
        sys.exit(f"no metadata.jsonl in {data_dir} -- run "
                 f"`python scripts/caption_corpus.py export` first")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{i}: {exc}")
        if "file_name" not in rec or "text" not in rec:
            sys.exit(f"{path}:{i}: needs both file_name and text")
        if not (data_dir / rec["file_name"]).exists():
            sys.exit(f"{path}:{i}: missing image {rec['file_name']}")
        rows.append(rec)
    if not rows:
        sys.exit(f"{path} is empty")
    return rows


def build_samples(data_dir: Path, rows: list[dict], resolution: int,
                  bucket: bool) -> list[Sample]:
    from PIL import Image

    buckets = make_buckets(resolution) if bucket else [(resolution, resolution)]
    samples = []
    for rec in rows:
        with Image.open(data_dir / rec["file_name"]) as img:
            size = img.size
        chosen = pick_bucket(size, buckets)
        # upscaling means the bucket asks for more pixels than the file has on
        # at least one axis after the cover-resize
        scale = max(chosen[0] / size[0], chosen[1] / size[1])
        samples.append(Sample(
            file_name=rec["file_name"],
            text=rec["text"],
            bucket=chosen,
            upscaled=scale > 1.0,
        ))
    return samples


def report_buckets(samples: list[Sample]) -> None:
    from collections import Counter

    counts = Counter(s.bucket for s in samples)
    up = sum(1 for s in samples if s.upscaled)
    print(f"  {len(counts)} bucket(s) in use, {up}/{len(samples)} images upscaled to fit")
    for (w, h), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {w:>4}x{h:<4} ar {w/h:.2f}  {n:>3} image(s)")


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

def cache_dataset(samples: list[Sample], data_dir: Path, pipe_parts: dict,
                  random_flip: bool, device, weight_dtype) -> None:
    """Fill in latents and text embeddings, then free the encoders.

    Latent *distributions* are cached rather than a single sample, so each epoch
    still draws a fresh latent -- the VAE's own noise stays as augmentation.
    """
    import torch
    from PIL import Image

    vae = pipe_parts["vae"]
    tokenizer = pipe_parts["tokenizer"]
    text_encoder = pipe_parts["text_encoder"]

    truncated = []
    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            with Image.open(data_dir / s.file_name) as raw:
                img = fit_to_bucket(raw.convert("RGB"), s.bucket)

            def encode(pil):
                arr = torch.from_numpy(
                    (np.asarray(pil, dtype=np.float32) / 127.5 - 1.0)
                ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype)
                dist = vae.encode(arr).latent_dist
                return (dist.mean.squeeze(0).cpu().to(torch.float16),
                        dist.std.squeeze(0).cpu().to(torch.float16))

            s.latent_mean, s.latent_std = encode(img)
            if random_flip:
                s.latent_mean_flip, s.latent_std_flip = encode(
                    img.transpose(Image.FLIP_LEFT_RIGHT))

            ids = tokenizer(s.text, padding="max_length", truncation=True,
                            max_length=CLIP_MAX_TOKENS, return_tensors="pt")
            # the caption card claims max 75 content tokens -- verify it here
            # rather than trusting it, because truncation is silent
            unbounded = tokenizer(s.text, truncation=False)["input_ids"]
            if len(unbounded) > CLIP_MAX_TOKENS:
                truncated.append((s.file_name, len(unbounded)))
            if TRIGGER not in s.text:
                print(f"  ! {s.file_name}: caption has no '{TRIGGER}' trigger")
            s.embedding = text_encoder(ids.input_ids.to(device))[0] \
                .squeeze(0).cpu().to(torch.float16)

            if i % 25 == 0 or i == len(samples):
                print(f"  cached {i}/{len(samples)}")

    if truncated:
        print(f"  ! {len(truncated)} caption(s) exceed {CLIP_MAX_TOKENS} CLIP tokens "
              f"and were truncated:")
        for name, n in truncated[:5]:
            print(f"      {name}: {n} tokens")

    # the encoders are done; give the UNet their VRAM back
    vae.to("cpu")
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    print(f"  cache built in {time.time() - t0:.0f}s, encoders moved to CPU")


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, step: int, unet, optimizer, lr_scheduler,
                    args, keep: int) -> Path:
    import torch
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    ckpt = out_dir / f"checkpoint-{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)

    peft_sd = get_peft_model_state_dict(unet)
    # inference-ready: pipe.load_lora_weights(ckpt) picks this up directly
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(ckpt),
        unet_lora_layers=convert_state_dict_to_diffusers(peft_sd),
        safe_serialization=True,
    )
    # resume-ready: optimiser moments matter, 8-bit Adam included
    torch.save({
        "step": step,
        "lora": peft_sd,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": vars(args),
    }, ckpt / "training_state.pt")

    existing = sorted(out_dir.glob("checkpoint-*"))
    if keep > 0 and len(existing) > keep:
        for old in existing[:-keep]:
            # the final weights are small; it is the optimiser state that fills
            # Drive, so prune whole directories rather than keeping stubs
            shutil.rmtree(old, ignore_errors=True)
            print(f"  pruned {old.name}")
    return ckpt


def find_resume(out_dir: Path, resume: str) -> Path | None:
    if resume in ("", None):
        return None
    if resume == "latest":
        found = sorted(out_dir.glob("checkpoint-*"))
        if not found:
            print(f"  --resume latest: no checkpoint in {out_dir}, starting fresh")
            return None
        return found[-1]
    path = Path(resume)
    if not path.exists():
        sys.exit(f"--resume {resume}: not found")
    return path


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def run_validation(pipe_parts, unet, args, step: int, out_dir: Path,
                   device, weight_dtype) -> None:
    """Sample the trigger prompt so run 1 answers the §5 'trigger does nothing' risk."""
    import torch
    from diffusers import StableDiffusionPipeline

    print(f"  validation at step {step}: {args.validation_prompt!r}")
    pipe = StableDiffusionPipeline(
        vae=pipe_parts["vae"].to(device),
        text_encoder=pipe_parts["text_encoder"].to(device),
        tokenizer=pipe_parts["tokenizer"],
        unet=unet,
        scheduler=pipe_parts["scheduler_inference"],
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    unet.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype,
                                         enabled=weight_dtype != torch.float32):
        images = pipe(
            [args.validation_prompt] * args.validation_images,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images
    unet.train()

    dest = out_dir / "samples"
    dest.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(dest / f"step{step:06d}_{i}.png")
    print(f"    wrote {len(images)} sample(s) to {dest}")

    # hand the VRAM straight back -- the UNet needs it for the next step
    pipe_parts["vae"].to("cpu")
    pipe_parts["text_encoder"].to("cpu")
    del pipe
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def require_gpu_stack():
    """Fail with the actual next action rather than a bare ModuleNotFoundError.

    The GPU dependencies are intentionally absent from the local environment --
    this machine has no CUDA device, so there is nothing to install them for.
    """
    missing = []
    for mod, pin in (("torch", "torch"), ("diffusers", "diffusers>=0.27"),
                     ("transformers", "transformers>=4.38"), ("peft", "peft>=0.10")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pin)
    if missing:
        sys.exit("missing GPU dependencies: " + ", ".join(missing) + "\n"
                 "This script needs a GPU -- run it in Colab or Kaggle via\n"
                 "  notebooks/milestone2_lora_colab.ipynb\n"
                 "or, in an environment that already has torch:\n"
                 "  pip install -r notebooks/requirements-gpu.txt")


def train(args) -> int:
    require_gpu_stack()
    import torch
    import torch.nn.functional as F
    from diffusers import (AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler,
                           UNet2DConditionModel)
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig, set_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    if not torch.cuda.is_available():
        sys.exit("no CUDA device. This script needs a GPU -- run it in Colab or "
                 "Kaggle (plans/PLAN.md §3 Milestone 2).")

    device = torch.device("cuda")
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                    "no": torch.float32}[args.mixed_precision]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print(f"base model: {args.base_model}")
    print(f"output    : {out_dir.resolve()}")
    print(f"gpu       : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ---- data ------------------------------------------------------------
    rows = load_metadata(data_dir)
    samples = build_samples(data_dir, rows, args.resolution, bucket=not args.no_bucket)
    print(f"\ndataset: {len(samples)} image(s) from {data_dir}")
    report_buckets(samples)

    # ---- model -----------------------------------------------------------
    print("\nloading base model")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=weight_dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", torch_dtype=weight_dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    pipe_parts = {
        "vae": vae, "tokenizer": tokenizer, "text_encoder": text_encoder,
        "scheduler_inference": DPMSolverMultistepScheduler.from_pretrained(
            args.base_model, subfolder="scheduler"),
    }

    print("\ncaching latents and text embeddings")
    cache_dataset(samples, data_dir, pipe_parts, args.random_flip, device, weight_dtype)

    # ---- lora ------------------------------------------------------------
    unet.add_adapter(LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGETS,
    ))
    if weight_dtype != torch.float32:
        # base weights stay fp16; the trainable adapters must be fp32 or the
        # updates vanish under the optimiser's epsilon
        cast_training_params(unet, dtype=torch.float32)

    params = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"\nlora: rank {args.rank}, alpha {args.lora_alpha}, "
          f"{n_trainable/1e6:.2f}M trainable params on {len(LORA_TARGETS)} "
          f"projection types (text encoder frozen)")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("  xformers attention enabled")
        except Exception as exc:
            print(f"  xformers unavailable ({exc}); using default attention")

    # ---- optimiser -------------------------------------------------------
    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            print("  8-bit Adam enabled")
        except ImportError:
            print("  bitsandbytes not installed; falling back to fp32 AdamW")
    optimizer = optimizer_cls(params, lr=args.learning_rate,
                              betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    # one image per forward pass -- a batch would have to share a bucket
    # resolution, and grad-accum gives the effective batch the plan asks for
    steps_per_epoch = math.ceil(len(samples) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # ---- resume ----------------------------------------------------------
    start_step = 0
    ckpt = find_resume(out_dir, args.resume)
    if ckpt:
        state = torch.load(ckpt / "training_state.pt", map_location="cpu")
        set_peft_model_state_dict(unet, state["lora"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        start_step = state["step"]
        print(f"\nresumed from {ckpt.name} at step {start_step}")

    # ---- loss log --------------------------------------------------------
    # appended and flushed every logging step: the curve is as much the
    # Milestone 2 deliverable as the weights, and Colab will kill this session
    loss_csv = out_dir / "loss.csv"
    new_log = not loss_csv.exists()
    log_fh = loss_csv.open("a", newline="", encoding="utf-8")
    logger = csv.writer(log_fh)
    if new_log:
        logger.writerow(["step", "epoch", "loss", "lr", "wall_seconds"])
        log_fh.flush()

    (out_dir / "config.json").write_text(
        json.dumps(vars(args) | {"trainable_params": n_trainable,
                                 "steps_per_epoch": steps_per_epoch}, indent=2),
        encoding="utf-8")

    # ---- loop ------------------------------------------------------------
    print(f"\ntraining to step {args.max_train_steps} "
          f"(~{args.max_train_steps/steps_per_epoch:.1f} epochs over {len(samples)} images, "
          f"effective batch {args.gradient_accumulation_steps})")

    unet.train()
    step = start_step
    t0 = time.time()
    running, running_n = 0.0, 0
    order: list[Sample] = []
    epoch = step // steps_per_epoch
    micro = 0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_train_steps:
        if not order:
            order = samples[:]
            random.shuffle(order)
        s = order.pop()

        # resample the latent from the cached distribution, and flip half the
        # time if --random-flip: the only two augmentations that are safe for a
        # style LoRA (colour jitter would corrupt the palette we are learning)
        flip = args.random_flip and s.latent_mean_flip is not None and random.random() < 0.5
        mean = (s.latent_mean_flip if flip else s.latent_mean).to(device, torch.float32)
        std = (s.latent_std_flip if flip else s.latent_std).to(device, torch.float32)
        latents = (mean + std * torch.randn_like(std)) * vae.config.scaling_factor
        latents = latents.unsqueeze(0).to(weight_dtype)
        encoder_hidden_states = s.embedding.unsqueeze(0).to(device, weight_dtype)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (1,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(noisy, timesteps, encoder_hidden_states).sample
        target = (noise if noise_scheduler.config.prediction_type == "epsilon"
                  else noise_scheduler.get_velocity(latents, noise, timesteps))
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        (loss / args.gradient_accumulation_steps).backward()
        running += loss.item()
        running_n += 1
        micro += 1

        if micro % args.gradient_accumulation_steps:
            continue

        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        epoch = step // max(steps_per_epoch, 1)

        if step % args.logging_steps == 0 or step == args.max_train_steps:
            avg = running / max(running_n, 1)
            logger.writerow([step, epoch, f"{avg:.6f}",
                             f"{lr_scheduler.get_last_lr()[0]:.3e}",
                             f"{time.time() - t0:.1f}"])
            log_fh.flush()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {step:>5}/{args.max_train_steps}  loss {avg:.4f}  "
                  f"lr {lr_scheduler.get_last_lr()[0]:.2e}  "
                  f"{(time.time()-t0)/max(step-start_step,1):.2f}s/step  "
                  f"peak {peak:.1f}GB")
            running, running_n = 0.0, 0

        if step % args.checkpointing_steps == 0 and step < args.max_train_steps:
            path = save_checkpoint(out_dir, step, unet, optimizer, lr_scheduler,
                                   args, args.checkpoints_total_limit)
            print(f"  checkpoint -> {path}")

        if (args.validation_steps and args.validation_prompt
                and step % args.validation_steps == 0):
            run_validation(pipe_parts, unet, args, step, out_dir, device, weight_dtype)

    # ---- finish ----------------------------------------------------------
    final = save_checkpoint(out_dir, step, unet, optimizer, lr_scheduler, args, 0)
    log_fh.close()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"  weights   {final / 'pytorch_lora_weights.safetensors'}")
    print(f"  loss curve {loss_csv}")
    already_sampled = (args.validation_steps
                       and step % args.validation_steps == 0)
    if args.validation_prompt and not already_sampled:
        run_validation(pipe_parts, unet, args, step, out_dir, device, weight_dtype)
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = ap.add_argument_group("data")
    g.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "train"),
                   help="directory holding the images and metadata.jsonl")
    g.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "lora"),
                   help="checkpoints, loss.csv and samples; point at Drive on Colab")
    g.add_argument("--resolution", type=int, default=512,
                   help="target area is resolution**2 (default 512; see module docstring "
                        "before raising this to 768)")
    g.add_argument("--no-bucket", action="store_true",
                   help="square centre-crop instead of aspect-ratio buckets (baseline run)")
    g.add_argument("--random-flip", action="store_true", default=True,
                   help="horizontal flip augmentation (default on)")
    g.add_argument("--no-random-flip", dest="random_flip", action="store_false")

    g = ap.add_argument_group("model")
    g.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="HF repo id. The original runwayml/stable-diffusion-v1-5 was "
                        "withdrawn; this is the canonical re-upload")
    g.add_argument("--rank", type=int, default=32,
                   help="LoRA rank (32 for style, per plans/PLAN.md; 16 is the ablation)")
    g.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA alpha (defaults to rank)")

    g = ap.add_argument_group("optimisation")
    g.add_argument("--learning-rate", type=float, default=1e-4)
    g.add_argument("--lr-scheduler", default="constant_with_warmup")
    g.add_argument("--lr-warmup-steps", type=int, default=100)
    g.add_argument("--max-train-steps", type=int, default=2500)
    g.add_argument("--gradient-accumulation-steps", type=int, default=4)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--mixed-precision", choices=["fp16", "bf16", "no"], default="fp16",
                   help="fp16 -- the T4 has no bf16")
    g.add_argument("--use-8bit-adam", action="store_true", default=True)
    g.add_argument("--no-8bit-adam", dest="use_8bit_adam", action="store_false")
    g.add_argument("--gradient-checkpointing", action="store_true", default=True)
    g.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    g.add_argument("--xformers", action="store_true", default=True)
    g.add_argument("--no-xformers", dest="xformers", action="store_false")
    g.add_argument("--seed", type=int, default=42)

    g = ap.add_argument_group("checkpointing and logging")
    g.add_argument("--checkpointing-steps", type=int, default=500,
                   help="Colab kills sessions; an uncheckpointed run is a lost day")
    g.add_argument("--checkpoints-total-limit", type=int, default=3,
                   help="0 keeps every checkpoint (optimiser state is ~100MB each)")
    g.add_argument("--logging-steps", type=int, default=25)
    g.add_argument("--resume", default="",
                   help="'latest' or a path to a checkpoint directory")
    g.add_argument("--validation-steps", type=int, default=500,
                   help="0 disables sampling during training")
    g.add_argument("--validation-prompt",
                   default=f"{TRIGGER}, Shakuntala writing a letter, seated under a tree",
                   help="tests the trigger -- the top risk in plans/PLAN.md §5")
    g.add_argument("--validation-images", type=int, default=2)

    args = ap.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = args.rank
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
