"""Milestones 3 + 4 -- text (and optional pose/edge reference) in, painting out.

    python src/generate.py "Yudhishthira weeping over the fallen Karna" --lora runs/r32/checkpoint-750
    python src/generate.py "Draupadi at the court" --lora runs/r32 --control pose.png --mode openpose

Needs a GPU. One pipeline is built once and reused: the LoRA is loaded onto the shared
UNet, and the ControlNet is swapped per call, so the T4 holds SD1.5 + one LoRA + two
ControlNets (~0.7 GB each) comfortably.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from expand import expand

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNETS = {
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "canny": "lllyasviel/control_v11p_sd15_canny",
}
# openpose: run the detector on a photo. canny: edge map. skeleton: the upload IS the
# control image (a stick-figure drawing or an already-rendered OpenPose skeleton) --
# the detector often finds nothing in a crude sketch, so this mode skips it.
MODES = ["openpose", "canny", "skeleton"]


class Painter:
    def __init__(self, lora: str | None, base: str = BASE, device: str = "cuda"):
        from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                               StableDiffusionControlNetPipeline,
                               StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)
        self.device = device
        if lora:
            lp = Path(lora)
            if not lp.exists():
                near = sorted(str(x) for x in lp.parent.glob("*"))[:10] if lp.parent.exists() else []
                raise FileNotFoundError(f"LoRA path not found: {lora}\nnearby: {near or 'parent folder missing too'}")
            if lp.is_dir() and not list(lp.glob("*.safetensors")):
                raise FileNotFoundError(f"no .safetensors inside {lora}; point at a checkpoint-N folder")        
        dt = torch.float16
        self._cn = {m: ControlNetModel.from_pretrained(r, torch_dtype=dt) for m, r in CONTROLNETS.items()}
        cn_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base, controlnet=self._cn["openpose"], torch_dtype=dt,
            safety_checker=None, requires_safety_checker=False).to(device)
        cn_pipe.scheduler = DPMSolverMultistepScheduler.from_config(cn_pipe.scheduler.config)
        if lora:
            cn_pipe.load_lora_weights(lora)  # lands on the UNet every pipeline below shares
        parts = {k: v for k, v in cn_pipe.components.items() if k != "controlnet"}
        self.cn_pipe = cn_pipe
        self.txt_pipe = StableDiffusionPipeline(**parts, requires_safety_checker=False)
        self.img_pipe = StableDiffusionImg2ImgPipeline(**parts, requires_safety_checker=False)
        cn_pipe.vae.enable_slicing()
        for p in (cn_pipe, self.txt_pipe, self.img_pipe):
            p.set_progress_bar_config(disable=True)
        self._pose = self._canny = None

    # --- control image -----------------------------------------------------------
    def control_image(self, ref: Image.Image, mode: str, size: tuple[int, int]) -> Image.Image:
        ref = ImageOps.fit(ref.convert("RGB"), size, Image.LANCZOS)
        if mode == "skeleton":
            return ref
        if mode == "canny":
            from controlnet_aux import CannyDetector
            self._canny = self._canny or CannyDetector()
            return self._canny(ref, low_threshold=100, high_threshold=200)
        from controlnet_aux import OpenposeDetector
        self._pose = self._pose or OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        return self._pose(ref)

    # --- generation --------------------------------------------------------------
    def check_tokens(self, *texts: str):
        """The guard from the risk register: refuse rather than silently truncate."""
        for t in texts:
            n = len(self.txt_pipe.tokenizer(t).input_ids)
            if n > 77:
                raise ValueError(f"prompt is {n} tokens; CLIP keeps 77 and would drop the tail: {t!r}")

    def paint(self, prompt: str, negative: str, ref: Image.Image | None = None,
              mode: str = "openpose", control_scale: float = 1.0, lora_scale: float = 1.0,
              size: tuple[int, int] = (512, 768), steps: int = 30, cfg: float = 7.5,
              seed: int = 42, hires: float = 1.5):
        """Returns (final image, control image or None). hires>1 adds an img2img pass."""
        self.check_tokens(prompt, negative)
        g = torch.Generator(self.device).manual_seed(seed)
        kw = dict(prompt=prompt, negative_prompt=negative, num_inference_steps=steps,
                  guidance_scale=cfg, generator=g, width=size[0], height=size[1],
                  cross_attention_kwargs={"scale": lora_scale})
        ctrl = None
        if ref is None:
            img = self.txt_pipe(**kw).images[0]
        else:
            ctrl = self.control_image(ref, mode, size)
            self.cn_pipe.controlnet = self._cn["canny" if mode == "canny" else "openpose"]
            img = self.cn_pipe(image=ctrl, controlnet_conditioning_scale=control_scale, **kw).images[0]
        if hires and hires > 1:
            # low-strength second pass: adds fine detail at 1.5x while the layout (and the
            # pose) stay put, and the LoRA keeps the brushwork consistent at the new size.
            big = img.resize((int(size[0] * hires) // 8 * 8, int(size[1] * hires) // 8 * 8), Image.LANCZOS)
            for k in ("width", "height"):
                kw.pop(k)
            kw["generator"] = torch.Generator(self.device).manual_seed(seed)
            img = self.img_pipe(image=big, strength=0.4, **kw).images[0]
        return img, ctrl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--lora", required=True, help="checkpoint dir or .safetensors from train_lora.py")
    ap.add_argument("--control", help="pose photo / stick figure / edge reference")
    ap.add_argument("--mode", choices=MODES, default="openpose")
    ap.add_argument("--no-llm", action="store_true", help="template expansion only")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--control-scale", type=float, default=1.0)
    ap.add_argument("--hires", type=float, default=1.5, help="upscale factor; 1 disables")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/painting.png")
    a = ap.parse_args()

    e = expand(a.scene, use_llm=not a.no_llm)
    print(f"[{e.source}] {e.tokens} tokens\n{e.prompt}\n" + (f"note: {e.note}\n" if e.note else ""))
    img, _ = Painter(a.lora).paint(
        e.prompt, e.negative, Image.open(a.control) if a.control else None, a.mode,
        a.control_scale, a.lora_scale, seed=a.seed, hires=a.hires)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out)
    print("saved", a.out, img.size)


if __name__ == "__main__":
    main()
