"""Milestone 3 -- prompt expansion.

    "Yudhishthira weeping over the fallen Karna on Kurukshetra, dramatic shadows, oil canvas"
        -> "rvrma, Yudhishthira, a scene from the Mahabharata, weeping over a fallen
            warrior, ..., battlefield with fallen chariots and banners"

The LoRA was trained on CONTENT-only captions behind the trigger `rvrma`
(caption_corpus.py, CAPTION_CARD.md), so the expanded prompt must look like those
captions: comma-separated content clauses, trigger first, no style words, <= 75
CLIP tokens. Style words the user typed ("oil canvas", "dramatic") are stripped for
the same reason the captions never contain them -- the trigger carries the style.

An LLM (Groq or Gemini, free tiers) does the expansion; a deterministic gazetteer
template is the fallback, so a rate limit or missing key never kills the demo.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the corpus tooling so the lint and the token budget cannot drift from training.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import caption_corpus as cc  # noqa: E402

TRIGGER = cc.TRIGGER
BUDGET = cc.CLIP_CONTENT_BUDGET  # 75: CLIP's 77 minus BOS/EOS

# Things we do NOT want, whatever the user typed. Short: it has its own 77-token cap.
NEGATIVE = ("photograph, lithograph, print, cartoon, anime, 3d render, text, watermark, "
            "modern clothing, deformed, bad anatomy, extra limbs, extra fingers, blurry")

# --------------------------------------------------------------------------- knowledge
# Period visual cues, in the register of the training captions. Keyed by lowercase
# needle (substring match, like the corpus gazetteer). Kept short on purpose: each
# entry costs tokens out of a 75-token budget.
FIGURE_CUES = {
    "yudhishthira": "Yudhishthira, king in white dhoti and gold crown, jewelled necklace, bare chest",
    "karna": "Karna, warrior in gold armour and earrings, long dark hair",
    "arjuna": "Arjuna, archer in dhoti and gold armlets, bow in hand, crown",
    "bhima": "Bhima, broad muscular warrior, gada mace, dhoti, gold armlets",
    "krishna": "Krishna, dark-skinned, yellow dhoti, peacock feather crown, flute",
    "draupadi": "Draupadi, woman in red sari with gold border, long loose hair, jewellery",
    "shakuntala": "Shakuntala, young woman in sari, flowers in hair, bare feet",
    "damayanti": "Damayanti, woman in silk sari, gold jewellery, swan beside her",
    "sita": "Sita, woman in simple sari with draped end over shoulder, gold bangles",
    "rama": "Rama, prince in dhoti, bow, crown, gold ornaments",
    "lakshmana": "Lakshmana, young warrior in dhoti with bow and quiver",
    "ravana": "Ravana, many-headed king in gold crowns, heavy jewellery",
    "hanuman": "Hanuman, monkey warrior, gada mace, red loincloth",
    "dushyanta": "Dushyanta, king in silk dhoti, gold crown, royal robes",
    "bhishma": "Bhishma, elderly warrior, white beard, white dhoti, arrows",
    "duryodhana": "Duryodhana, king in gold crown, armour, red dhoti",
    "kunti": "Kunti, mother in white sari, grief, hands clasped",
    "gandhari": "Gandhari, queen in sari, blindfold of cloth",
    "nala": "Nala, king in silk dhoti, gold crown, jewellery",
    "saraswati": "Saraswati, goddess in white sari, veena, lotus seat",
    "lakshmi": "Lakshmi, goddess in red and gold sari, lotus flowers",
}
EPIC_CLAUSE = {  # from the corpus gazetteer; nothing about the epic is invented here
    "mahabharata": cc.EPIC_CLAUSE["Mahabharata"],
    "ramayana": cc.EPIC_CLAUSE["Ramayana"],
}
FIGURE_EPIC = {  # which epic a known figure belongs to
    **dict.fromkeys(["yudhishthira", "karna", "arjuna", "bhima", "krishna", "draupadi",
                     "dushyanta", "bhishma", "duryodhana", "kunti", "gandhari", "nala",
                     "damayanti", "kurukshetra", "pandava", "kaurava"], "mahabharata"),
    **dict.fromkeys(["rama", "sita", "lakshmana", "ravana", "hanuman", "lanka"], "ramayana"),
}
SETTING_CUES = {  # trigger word -> background clause, in caption register
    "kurukshetra": "battlefield with fallen chariots, broken arrows and banners, smoke",
    "battlefield": "battlefield with fallen chariots, broken arrows and banners, smoke",
    "forest": "dense forest, tall trees, dappled ground",
    "palace": "palace hall, carved stone columns, draped curtains",
    "court": "court hall, stone columns, draped curtains",
    "river": "river bank, distant hills, clear sky",
    "garden": "garden, flowering trees, stone steps",
    "temple": "temple courtyard, stone columns, oil lamps",
    "swayamvara": "hall of seated kings, stone columns, garlands",
}
POSE_CUES = {  # emotion/action -> pose clause
    "weep": "kneeling, head bowed, hands near face",
    "grie": "kneeling, head bowed, hands near face",
    "mourn": "kneeling, head bowed, hands near face",
    "fight": "standing firm, bow drawn, facing forward",
    "battle": "standing firm, bow drawn, facing forward",
    "pray": "standing, hands joined, looking upward",
    "writ": "seated, looking down at a letter",
    "garland": "standing, holding a garland, looking to the side",
}

# --------------------------------------------------------------------------- helpers
_SPLIT = re.compile(r"\s*,\s*")


def strip_style(text: str) -> tuple[str, list[str]]:
    """Remove words the caption lint bans. Returns (clean text, removed words)."""
    masked = text
    for ph in cc.CONTENT_EXCEPTIONS:  # "oil lamp" is a prop, keep it
        masked = re.sub(re.escape(ph), ph.replace(" ", "_"), masked, flags=re.I)
    removed = []
    for w in cc.BANNED_WORDS:
        masked, n = re.subn(rf"\b{re.escape(w)}\b", "", masked, flags=re.I)
        if n:
            removed.append(w)
    clauses = [re.sub(r"\s+", " ", c).strip() for c in _SPLIT.split(masked.replace("_", " "))]
    return ", ".join(c for c in clauses if c), removed


def fit_budget(clauses: list[str]) -> str:
    """Join clauses behind the trigger, dropping whole trailing clauses until it fits.

    Clauses are ordered most-important-first, so truncation here is deliberate and
    clean, unlike CLIP's silent mid-phrase cut.
    """
    clauses = [c for c in clauses if c]
    while True:
        text = ", ".join([TRIGGER] + clauses)
        if cc.count_tokens(text)[0] <= BUDGET or len(clauses) <= 1:
            return text
        clauses.pop()


def _entities(text: str) -> list[str]:
    low = text.lower()
    return [k for k in FIGURE_CUES if k in low]


# --------------------------------------------------------------------------- fallback
def template_expand(scene: str) -> str:
    """Deterministic expansion. No network, no model."""
    low = scene.lower()
    ents = _entities(scene)
    epic = next((EPIC_CLAUSE[FIGURE_EPIC[k]] for k in [*ents, *low.split()]
                 if FIGURE_EPIC.get(k) in EPIC_CLAUSE), None)
    clauses = [scene.strip().rstrip(".")]  # the user's own scene leads: it is the content
    # the scene already names them, so keep only the visual cue, not the repeated name
    clauses += [FIGURE_CUES[k].split(", ", 1)[1] for k in ents[:2]]
    if epic:
        clauses.append(epic)
    for table in (POSE_CUES, SETTING_CUES):
        hit = next((v for k, v in table.items() if k in low), None)
        if hit:
            clauses.append(hit)
    clauses.append("centered figures, balanced composition")
    return fit_budget(clauses)


# --------------------------------------------------------------------------- LLM
SYSTEM = f"""You write image-generation prompts for a model fine-tuned on 19th-century Indian \
academic realist paintings. Output ONE line: comma-separated visual clauses, no sentences.
Order: the main figures (named), epic context, pose and gaze, clothing and ornaments, props, \
setting, composition. Be concrete: period armour, dhoti and sari folds, jewellery, banners.
Hard rules: at most 45 words. NEVER use style or medium words (painting, oil, canvas, art, \
style, lighting, dramatic, soft, rich, detailed, realistic, beautiful, masterpiece). Do not \
write the trigger token. Output the line only."""


def _post(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def llm_expand(scene: str, provider: str | None = None) -> str:
    """Raw LLM line for `scene`. Raises on any failure -- the caller falls back."""
    hints = "; ".join(FIGURE_CUES[k] for k in _entities(scene)[:3])
    user = f"Scene: {scene}" + (f"\nKnown figures: {hints}" if hints else "")
    provider = provider or ("groq" if os.environ.get("GROQ_API_KEY") else "gemini")
    if provider == "groq":
        out = _post("https://api.groq.com/openai/v1/chat/completions",
                    {"model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                     "temperature": 0.4, "max_tokens": 120,
                     "messages": [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": user}]},
                    {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        return out["choices"][0]["message"]["content"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    out = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"systemInstruction": {"parts": [{"text": SYSTEM}]},
                 "contents": [{"parts": [{"text": user}]}],
                 "generationConfig": {"temperature": 0.4, "maxOutputTokens": 120}},
                {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    return out["candidates"][0]["content"]["parts"][0]["text"]


# --------------------------------------------------------------------------- public API
@dataclass
class Expansion:
    prompt: str
    negative: str = NEGATIVE
    tokens: int = 0
    source: str = "template"           # "llm" | "template"
    stripped: list[str] = field(default_factory=list)  # style words removed from user text
    note: str = ""                     # why the LLM was skipped, if it was


def expand(scene: str, use_llm: bool = True, provider: str | None = None) -> Expansion:
    scene, stripped = strip_style(scene)
    if not scene:
        raise ValueError("scene is empty after removing style words")
    source, note, clauses = "template", "", None
    if use_llm and (os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        try:
            raw = llm_expand(scene, provider).strip().strip('"').splitlines()[0]
            raw = raw.removeprefix(TRIGGER).lstrip(", ")
            clean, more = strip_style(raw)  # the LLM disobeys sometimes; the lint doesn't
            stripped += more
            # the user's wording can be lost in expansion, so keep it as the lead clause
            clauses = [scene] + _SPLIT.split(clean)
            source = "llm"
        except Exception as e:  # rate limit, timeout, bad JSON, no network: all the same
            note = f"LLM failed ({type(e).__name__}); used template"
    elif use_llm:
        note = "no GROQ_API_KEY / GEMINI_API_KEY set; used template"
    prompt = fit_budget(clauses) if clauses else template_expand(scene)
    return Expansion(prompt, NEGATIVE, cc.count_tokens(prompt)[0], source, sorted(set(stripped)), note)


if __name__ == "__main__":
    # self-check: network-free
    e = expand("Yudhishthira weeping over the fallen body of Karna on the battlefield of "
               "Kurukshetra, dramatic shadows, oil canvas", use_llm=False)
    print(e)
    assert e.prompt.startswith(TRIGGER + ",") and e.tokens <= BUDGET
    assert not cc.lint_caption(e.prompt)[0], cc.lint_caption(e.prompt)
    assert {"oil", "canvas", "dramatic"} <= set(e.stripped)
    assert "Karna" in e.prompt and "Mahabharata" in e.prompt and "battlefield" in e.prompt
    assert cc.count_tokens(NEGATIVE)[0] <= BUDGET
    long = fit_budget(["word " * 20] * 6)
    assert cc.count_tokens(long)[0] <= BUDGET and long.startswith(TRIGGER)
    assert "oil lamp" in strip_style("an oil lamp, oil painting")[0]
    print("ok")
