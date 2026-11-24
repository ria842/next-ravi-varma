"""Milestone 4 -- Gradio panel: narrative + pose reference in, painting out.

    python app/app.py --lora runs/r32-750-epicrealism/checkpoint-000500 --share

Step 1 "Expand prompt" shows (and lets you edit) what the model will actually see;
step 2 "Paint" generates. Paint expands for you if you skip step 1.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr

from expand import NEGATIVE, expand
from generate import BASE, MODES, Painter

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--base", default=BASE, help="must be the base the LoRA was trained on")
ap.add_argument("--share", action="store_true", help="public gradio.live link (use in Colab)")
args = ap.parse_args()

painter = Painter(args.lora, args.base)
EXAMPLE = ("Yudhishthira weeping over the fallen body of Karna on the battlefield of "
           "Kurukshetra, dramatic shadows, oil canvas")


def do_expand(scene, use_llm):
    e = expand(scene, use_llm)
    info = f"{e.source} - {e.tokens}/75 tokens"
    if e.stripped:
        info += f" - removed style words (the trigger carries the style): {', '.join(e.stripped)}"
    return e.prompt, info + (f"\n{e.note}" if e.note else "")


def do_paint(scene, final, use_llm, ref, mode, control_scale, lora_scale, hires, seed, steps):
    if not final.strip():
        final, _ = do_expand(scene, use_llm)
    try:
        img, ctrl = painter.paint(final, NEGATIVE, ref, mode, control_scale,
                                  lora_scale, steps=int(steps), seed=int(seed), hires=hires)
    except ValueError as e:  # token guard
        raise gr.Error(str(e))
    return img, ctrl, final


with gr.Blocks(title="The Next Ravi Varma") as demo:
    gr.Markdown("# The Next Ravi Varma\nDescribe a scene from the epics. Optionally upload a "
                "pose reference so the figures keep Varma's balanced, static staging.")
    with gr.Row():
        with gr.Column():
            scene = gr.Textbox(label="Narrative prompt", value=EXAMPLE, lines=3)
            use_llm = gr.Checkbox(True, label="Expand with LLM (falls back to template)")
            expand_btn = gr.Button("1. Expand prompt")
            final = gr.Textbox(label="Final prompt (editable)", lines=4)
            info = gr.Markdown()
            ref = gr.Image(type="pil", label="Pose / composition reference (optional)")
            mode = gr.Radio(MODES, value="openpose", label="Reference is a...",
                            info="openpose: photo of a person · canny: any image's edges · "
                                 "skeleton: a stick-figure drawing or ready-made OpenPose map")
            with gr.Accordion("Advanced", open=False):
                control_scale = gr.Slider(0, 2, 1.0, step=0.05, label="Structure strength")
                lora_scale = gr.Slider(0, 1.5, 0.8, step=0.05, label="Varma style strength")
                hires = gr.Slider(1, 2, 1.5, step=0.25, label="Upscale (1 = off)")
                seed = gr.Number(42, label="Seed", precision=0)
                steps = gr.Slider(15, 50, 30, step=1, label="Steps")
            paint_btn = gr.Button("2. Paint", variant="primary")
        with gr.Column():
            out = gr.Image(label="Painting")
            ctrl = gr.Image(label="Structure map actually used")

    expand_btn.click(do_expand, [scene, use_llm], [final, info])
    paint_btn.click(do_paint, [scene, final, use_llm, ref, mode, control_scale, lora_scale,
                               hires, seed, steps], [out, ctrl, final])

demo.queue().launch(share=args.share)
