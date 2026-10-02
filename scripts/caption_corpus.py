#!/usr/bin/env python3
"""
Caption the Raja Ravi Varma corpus for style-LoRA training.

Pipeline
--------
1. draft  : BLIP-2 answers a fixed three-question template per image (figures and
            clothing / pose and gaze / background), and the answers are assembled
            into one caption behind the trigger token. Needs a GPU -- run it in
            Colab or Kaggle, not on the local machine.
2. review : human correction pass. Resumable, shardable across teammates, and it
            refuses to save a caption that fails the style lint.
3. lint   : standalone validation over every caption -- banned style words, CLIP
            token budget, trigger presence, coverage.
4. export : emit `metadata.jsonl` in the layout `diffusers` expects, split into
            train/ and holdout/ trees ready to zip and upload.

THE ONE RULE (plans/PLAN.md §1.3)
---------------------------------
Captions describe CONTENT, never STYLE. If a caption says "oil painting" or "soft
lighting", the LoRA binds Varma's style to those common English tokens instead of
to the trigger, and the trigger does nothing at inference. Style words are a hard
error here, not a warning -- see BANNED_WORDS.

Captions live in data/captions.jsonl, deliberately NOT in manifest.jsonl, because
scrape_commons.py rewrites the manifest wholesale on every run and would discard
the human correction pass with it.

Usage
-----
    python scripts/caption_corpus.py draft --limit 10      # GPU: smoke test
    python scripts/caption_corpus.py draft                 # GPU: all 120
    python scripts/caption_corpus.py review --shard 1/3    # local: teammate A
    python scripts/caption_corpus.py lint
    python scripts/caption_corpus.py export
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

# --- Trigger -----------------------------------------------------------------
# A rare, non-semantic token (plans/PLAN.md §1.4). "ravi varma" would tokenise
# into common subwords that already carry meaning, so the style would bleed into
# them instead of binding here.
TRIGGER = "rvrma"

# SD 1.5's CLIP text encoder hard-truncates at 77 tokens including BOS and EOS.
# Anything past 75 content tokens is silently discarded during training.
CLIP_MAX_TOKENS = 77
CLIP_CONTENT_BUDGET = CLIP_MAX_TOKENS - 2

# Calibrated against the real CLIP tokeniser over this corpus: the ratio of true
# BPE tokens to pre-tokens averages 1.07 and peaks at 1.18. 1.2 never
# underestimated on any of the 100 captions, which is the direction that matters
# -- underestimating means silent truncation, the failure this guard exists to
# prevent. It costs roughly 7 tokens of headroom, so install transformers for
# exact counts whenever the budget is tight.
PRETOKEN_FACTOR = 1.2


# ---------------------------------------------------------------------------
# Style lint
#
# BANNED is fatal: every one of these is a common English token that the LoRA
# would happily bind Varma's style to, leaving the trigger inert. SOFT is
# advisory -- meta-language about the image rather than content in it.
# ---------------------------------------------------------------------------

BANNED_WORDS = [
    # medium and process
    "paint", "paints", "painted", "painting", "paintings", "painter",
    "oil", "oils", "canvas", "artwork", "art", "artistic", "drawing", "drawn",
    "sketch", "illustration", "lithograph", "engraving", "watercolour",
    "watercolor", "brushstroke", "brushstrokes", "brushwork", "brush", "palette",
    "photo", "photograph", "photography", "image", "picture",
    # style and quality language
    "style", "styled", "stylish", "stylised", "stylized", "aesthetic",
    "masterpiece", "beautiful", "beauty", "gorgeous", "stunning", "exquisite",
    "detailed", "elegant", "lovely", "graceful",
    "realistic", "realism", "photorealistic", "lifelike", "hyperrealistic",
    # light and colour treatment
    "lighting", "chiaroscuro", "shading", "shaded", "luminous", "glowing",
    "vibrant", "rich", "soft", "muted", "dramatic", "atmospheric",
    # movements
    "renaissance", "baroque", "academic", "impressionist", "romanticism",
    "victorian", "classical",
]

SOFT_WORDS = [
    # Ambiguous rather than banned: "warm clothing" and "printed fabric" are
    # content, "warm glow" and "print" as a medium are not. The reviewer judges.
    "warm", "print", "printed",
    "depicts", "depicting", "depicted", "shows", "showing", "features",
    "featuring", "portrays", "portraying", "scene of", "view of", "close up",
]

# Phrases where a banned word is the name of a real object in the painting. An
# oil LAMP is a prop; an oil PAINTING is the medium. Masked out before linting.
CONTENT_EXCEPTIONS = [
    "oil lamp", "oil lamps", "oil wick", "canvas bag", "canvas sack",
]

# BLIP-2 tics. "arafed" is a well-known artefact of its caption training data and
# means nothing; the rest are framing prefixes that add no content.
NOISE_PATTERNS = [
    r"^\s*ara[fv]ed\s+",
    r"^\s*(?:there\s+is|there\s+are|this\s+is|it\s+is|we\s+see)\s+",
    r"^\s*(?:an?\s+)?(?:oil\s+)?(?:painting|portrait|picture|image|photo|drawing)\s+(?:of|showing|depicting)\s+",
    r"^\s*(?:in\s+)?(?:this|the)\s+(?:painting|image|picture|photo|scene)\s*,?\s*",
    r"\barafed\b",
]


# ---------------------------------------------------------------------------
# Entity gazetteer
#
# The Commons category tree is far thinner than plans/PLAN.md §2 assumes: of the
# 120 kept files only 27 carry an epic label, and "Ahalya by Ravi Varma" is filed
# under nothing more specific than "women of India". The TITLES, however, name
# the figure almost every time -- so names are recovered from the title, with the
# categories as a secondary source.
#
# Matching is substring, not word-boundary, because Commons titles run names
# together: "Draupadikichaka", "Draupadisairendri", "Draupadisimhika1b".
#
# These proper nouns are rare tokens too, so in principle they could compete with
# the trigger for the style. In practice each appears on 1-5 images while the
# trigger appears on all 100, so the trigger dominates by a wide margin.
# ---------------------------------------------------------------------------

FIGURES: list[tuple[str, str]] = [
    # (lowercase needle matched against title+categories, canonical caption form)
    ("shakuntala", "Shakuntala"), ("sakunthala", "Shakuntala"), ("sakuntala", "Shakuntala"),
    ("draupadi", "Draupadi"), ("sairandhri", "Sairandhri"), ("sairendri", "Sairandhri"),
    ("damayant", "Damayanti"), ("simhika", "Simhika"), ("kichaka", "Keechaka"),
    ("keechaka", "Keechaka"), ("duryodhana", "Duryodhana"), ("arjuna", "Arjuna"),
    ("subhadra", "Subhadra"), ("shantanu", "Shantanu"), ("satyavati", "Satyavati"),
    ("ganga", "Ganga"), ("yasodha", "Yashoda"), ("yashoda", "Yashoda"),
    ("krishna", "Krishna"), ("kamsa", "Kamsa"), ("radha", "Radha"),
    ("rugmangada", "Rugmangada"), ("mohini", "Mohini"), ("nala", "Nala"),
    ("menaka", "Menaka"), ("vishwamitra", "Vishvamitra"), ("vishvamitra", "Vishvamitra"),
    ("dushyanta", "Dushyanta"), ("ahalya", "Ahalya"), ("kaikeyi", "Kaikeyi"),
    ("ravana", "Ravana"), ("sita", "Sita"), ("jatayu", "Jatayu"),
    ("jathayu", "Jatayu"), ("meghanada", "Meghanada"), ("tilottama", "Tilottama"),
    ("kadambari", "Kadambari"), ("saraswati", "Saraswati"), ("lakshmi", "Lakshmi"),
    ("shiva", "Shiva"), ("tara", "Tara"),
]

# Longer words that merely CONTAIN a figure's name. Substring matching is what
# rescues the run-together Commons titles ("Draupadikichaka"), but it also turns
# a sitter called Sethu Lakshmi Bayi into the goddess Lakshmi and Princess
# Tarabai of Baroda into Tara of the Ramayana. These are cut out first.
NAME_FALSE_FRIENDS = [
    "lakshmipuram", "lakshmi bayi", "lakshmi vilas",   # a palace and a maharani
    "krishnaraja", "krishnaraj", "krishna raja",       # the Wadiyar kings
    "tarabai", "tharabai",                             # Princess Tarabai of Baroda
    "sitar",                                           # the instrument, not Sita
]

EPIC_CLAUSE = {
    "Mahabharata": "a scene from the Mahabharata",
    "Ramayana": "a scene from the Ramayana",
    "Abhijnanashakuntalam": "a scene from the Abhijnanashakuntalam",
}

# Demographic categories deliberately produce NO clause. BLIP-2 already opens with
# "a woman ..."; prepending "a woman of India" only burns tokens and teaches the
# model a phrase no user will ever type.
DEMOGRAPHIC_CATEGORIES = re.compile(
    r"^(portraits? of )?(women|men|children|couples) of (india|kerala)$", re.I
)

# Title noise: attribution, accession numbers, source collections, bare years.
TITLE_NOISE = [
    r"\bby\s+raja\s+ravi\s+varma\b", r"\braja\s+ravi\s+varma\b", r"\bravi\s+varma\b",
    r"\bvarma\b", r"\brrv\b", r"google art project", r"royal collection",
    r"rcin\s*\d+", r"wellcome\s*[a-z]?\d+", r"va ac \d+[a-z]*\d*",
    r"\(\s*1[89]\d{2}\s*(?:-\s*\d+)?\s*\)", r"\b1[89]\d{2}\b",
]


# ---------------------------------------------------------------------------
# The rigid template
#
# One free-form "describe this" call gives BLIP-2 room to editorialise about mood
# and beauty, which is exactly what the lint then rejects. Three narrow questions
# keep every answer on content, and each maps to one clause of the caption.
# ---------------------------------------------------------------------------

QUESTIONS: list[tuple[str, str]] = [
    ("figures", "Question: How many people are in this image, and what clothing and "
                "jewellery are they wearing? Answer:"),
    ("pose",    "Question: What are the people doing with their hands, and which "
                "direction are they facing? Answer:"),
    ("setting", "Question: What objects and surroundings are behind the people? Answer:"),
]


@dataclass
class Caption:
    """One image and its caption at whatever stage it has reached."""
    slug: str
    file: str
    title: str = ""
    split: str = ""
    subjects: list[str] = field(default_factory=list)
    epic: str | None = None
    entity: str = ""            # entity clause derived from title/categories
    parts: dict = field(default_factory=dict)   # raw per-question BLIP-2 answers
    draft: str = ""             # assembled machine caption, never overwritten by review
    caption: str = ""           # the caption actually used; human-editable
    reviewed: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------

def strip_noise(text: str) -> str:
    """Remove BLIP-2's framing tics and collapse whitespace."""
    out = text.strip()
    for _ in range(3):                      # prefixes nest: "this is a painting of ..."
        before = out
        for pattern in NOISE_PATTERNS:
            out = re.sub(pattern, " ", out, flags=re.I)
        out = out.strip(" ,.;:")
        if out == before:
            break
    return re.sub(r"\s+", " ", out).strip(" ,.;:")


def dedupe_clauses(clauses: list[str]) -> list[str]:
    """BLIP-2 repeats itself across questions; drop clauses already covered."""
    kept: list[str] = []
    for clause in clauses:
        clause = clause.strip(" ,.;:")
        if not clause:
            continue
        norm = re.sub(r"[^a-z0-9 ]", "", clause.lower())
        if any(norm in re.sub(r"[^a-z0-9 ]", "", k.lower()) for k in kept):
            continue
        kept = [k for k in kept if re.sub(r"[^a-z0-9 ]", "", k.lower()) not in norm]
        kept.append(clause)
    return kept


def count_tokens(text: str) -> tuple[int, bool]:
    """CLIP token count. Returns (count, exact) -- exact is False on the fallback.

    The real tokeniser only exists where transformers is installed, which is not
    the local machine. The fallback deliberately over-estimates: BPE splits proper
    nouns like "Sairandhri" into several tokens.
    """
    tokeniser = _clip_tokeniser()
    if tokeniser is not None:
        return len(tokeniser.tokenize(text)), True
    return int(round(len(_clip_pretokens(text)) * PRETOKEN_FACTOR)), False


# CLIP's BPE pre-tokeniser: letter runs, single digits, punctuation runs. Every
# pre-token becomes at least one BPE token, so the count is a hard lower bound.
def _clip_pretokens(text: str) -> list[str]:
    text = text.lower()
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif c.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            out.append(text[i:j]); i = j
        elif c.isdigit():
            out.append(c); i += 1          # [\p{N}] is ONE digit per token
        else:
            j = i
            while j < len(text) and not text[j].isspace() and not text[j].isalnum():
                j += 1
            out.append(text[i:j]); i = j
    return out


_TOKENISER_CACHE: list = []


def _clip_tokeniser():
    if _TOKENISER_CACHE:
        return _TOKENISER_CACHE[0]
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    except Exception:
        tok = None
    _TOKENISER_CACHE.append(tok)
    return tok


def lint_caption(text: str) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one caption."""
    errors, warnings = [], []
    low = text.lower()

    if not low.startswith(TRIGGER):
        errors.append(f"does not start with the trigger token '{TRIGGER}'")

    body = low[len(TRIGGER):] if low.startswith(TRIGGER) else low
    for phrase in CONTENT_EXCEPTIONS:
        body = body.replace(phrase, " ")
    for word in BANNED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", body):
            errors.append(f"banned style word: '{word}'")
    for word in SOFT_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", body):
            warnings.append(f"meta-language: '{word}'")

    n, exact = count_tokens(text)
    if n > CLIP_CONTENT_BUDGET:
        errors.append(
            f"{n} CLIP tokens{'' if exact else ' (estimated)'} "
            f"> {CLIP_CONTENT_BUDGET}; the tail would be silently truncated"
        )
    elif n > CLIP_CONTENT_BUDGET - 10:
        warnings.append(f"{n} tokens, close to the {CLIP_CONTENT_BUDGET} limit")

    if len(body.split()) < 6:
        errors.append("too short to be a useful caption")
    return sorted(set(errors)), sorted(set(warnings))


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def clean_title(title: str) -> str:
    out = re.sub(r"^File:", "", title)
    out = re.sub(r"\.(jpg|jpeg|png|tif|tiff)$", "", out, flags=re.I)
    for pattern in TITLE_NOISE:
        out = re.sub(pattern, " ", out, flags=re.I)
    return re.sub(r"[\s,\-_]+", " ", out).strip(" ,-")


def entity_clause(rec: dict) -> str:
    """Named figures plus epic context, as a caption fragment.

    Ground truth a VLM cannot produce: BLIP-2 will never guess "Shakuntala".
    """
    haystack = " ".join([rec.get("title", "")] + rec.get("categories", []) + rec.get("subjects", [])).lower()
    for phrase in NAME_FALSE_FRIENDS:
        haystack = haystack.replace(phrase, " ")

    names: list[str] = []
    for needle, canonical in FIGURES:
        if needle in haystack and canonical not in names:
            names.append(canonical)

    parts: list[str] = []
    if names:
        parts.append(" and ".join(names[:3]))

    epic = rec.get("epic")
    if epic in EPIC_CLAUSE:
        parts.append(EPIC_CLAUSE[epic])
    else:
        for subject in rec.get("subjects", []):
            if DEMOGRAPHIC_CATEGORIES.match(subject):
                continue
            if subject in EPIC_CLAUSE:
                parts.append(EPIC_CLAUSE[subject])
                break
    return ", ".join(parts)


def assemble(entity: str, parts: dict) -> str:
    """Trigger, then entity ground truth, then the VLM's content clauses."""
    # Split at comma granularity first: the three answers overlap heavily ("two
    # women, one in a red sari" / "two women, one holding a garland"), and whole
    # answers never compare equal even when half of each is the same phrase.
    clauses: list[str] = []
    for key, _ in QUESTIONS:
        answer = strip_noise(parts.get(key, ""))
        clauses.extend(part for part in answer.split(",") if part.strip())
    clauses = dedupe_clauses(clauses)
    body = ", ".join(([entity] if entity else []) + clauses)
    caption = f"{TRIGGER}, {body}" if body else TRIGGER

    # Trim from the tail until it fits rather than letting CLIP truncate silently.
    while count_tokens(caption)[0] > CLIP_CONTENT_BUDGET:
        bits = caption.rsplit(",", 1)
        if len(bits) == 1:
            break
        caption = bits[0]
    return caption.strip(" ,")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def load_manifest(data_dir: Path) -> list[dict]:
    path = data_dir / "manifest.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found -- run scripts/scrape_commons.py first.")
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in recs if r.get("status") == "kept"]


def captions_path(data_dir: Path) -> Path:
    return data_dir / "captions.jsonl"


def load_captions(data_dir: Path) -> dict[str, Caption]:
    path = captions_path(data_dir)
    if not path.exists():
        return {}
    out: dict[str, Caption] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            out[data["slug"]] = Caption(**data)
    return out


def save_captions(data_dir: Path, store: dict[str, Caption]) -> None:
    """Atomic write -- an interrupted review pass must not truncate the file."""
    path = captions_path(data_dir)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for slug in sorted(store):
            fh.write(json.dumps(asdict(store[slug]), ensure_ascii=False) + "\n")
    tmp.replace(path)


def image_for(rec: dict, images_dir: Path) -> Path | None:
    local = rec.get("local_path", "")
    if local:
        candidate = images_dir / Path(local).name
        if candidate.exists():
            return candidate
        if Path(local).exists():
            return Path(local)
    return None


def sync_store(data_dir: Path, images_dir: Path) -> dict[str, Caption]:
    """Seed the store from the manifest without disturbing existing captions."""
    store = load_captions(data_dir)
    for rec in load_manifest(data_dir):
        path = image_for(rec, images_dir)
        if path is None:
            continue
        slug = path.stem
        entry = store.get(slug)
        if entry is None:
            entry = Caption(slug=slug, file=path.name)
            store[slug] = entry
        # Manifest-derived fields are refreshed; human text never is.
        entry.file = path.name
        entry.title = clean_title(rec.get("title", ""))
        entry.split = rec.get("split", "")
        entry.subjects = rec.get("subjects", [])
        entry.epic = rec.get("epic")
        if not entry.reviewed:
            entry.entity = entity_clause(rec)
    return store


def select(store: dict[str, Caption], shard: str | None, split: str | None) -> list[Caption]:
    items = [store[s] for s in sorted(store)]
    if split:
        items = [c for c in items if c.split == split]
    if shard:
        try:
            index, total = (int(x) for x in shard.split("/"))
            assert 1 <= index <= total
        except (ValueError, AssertionError):
            sys.exit(f"--shard must look like 1/3, got {shard!r}")
        items = [c for i, c in enumerate(items) if i % total == index - 1]
    return items


# ---------------------------------------------------------------------------
# draft -- BLIP-2, GPU only
# ---------------------------------------------------------------------------

def cmd_draft(args) -> int:
    try:
        import torch
        from PIL import Image
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
    except ImportError as exc:
        sys.exit(
            f"missing dependency: {exc.name}\n"
            "Drafting needs a GPU box. On Colab or Kaggle:\n"
            "    pip install torch transformers accelerate Pillow"
        )

    store = sync_store(args.out, args.images)
    todo = [c for c in select(store, args.shard, None)
            if args.force or (not c.caption and not c.reviewed)]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Nothing to draft. Use --force to redraft unreviewed captions.")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible. BLIP-2 on CPU is roughly a minute per image.",
              file=sys.stderr)
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"loading {args.model} on {device}")
    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    for i, entry in enumerate(todo, 1):
        path = args.images / entry.file
        if not path.exists():
            print(f"  ! missing {path}", file=sys.stderr)
            continue
        image = Image.open(path).convert("RGB")

        for key, question in QUESTIONS:
            inputs = processor(images=image, text=question, return_tensors="pt").to(device, dtype)
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=3,
                    repetition_penalty=1.4,   # BLIP-2 loops on "and a red sari and a red sari"
                    length_penalty=0.9,
                )
            entry.parts[key] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        entry.draft = assemble(entry.entity, entry.parts)
        entry.caption = entry.draft
        entry.reviewed = False

        errors, _ = lint_caption(entry.caption)
        flag = "  [LINT]" if errors else ""
        print(f"  [{i}/{len(todo)}] {entry.file}{flag}\n      {entry.caption}")
        if i % 10 == 0:
            save_captions(args.out, store)

    save_captions(args.out, store)
    print(f"\nDrafts written to {captions_path(args.out)}")
    print("Next: python scripts/caption_corpus.py review --shard 1/3")
    return 0


# ---------------------------------------------------------------------------
# import -- adopt captions produced elsewhere
# ---------------------------------------------------------------------------

def inject_entity(text: str, entity: str) -> tuple[str, bool]:
    """Put the ground-truth entity clause straight after the trigger.

    A VLM cannot produce these: it sees a woman leaning on a branch, not Ahalya.
    Skipped when the caption already names the figure, and skipped rather than
    truncated if it would push the caption past the CLIP budget.
    """
    if not entity:
        return text, False
    names = [n for n in entity.split(",")[0].split(" and ") if n[:1].isupper()]
    if names and any(n.lower() in text.lower() for n in names):
        return text, False

    body = text[len(TRIGGER):].lstrip(" ,") if text.lower().startswith(TRIGGER) else text

    # Degrade gracefully rather than losing the name entirely. The epic clause is
    # the expensive half -- "a scene from the Abhijnanashakuntalam" alone is 12
    # CLIP tokens -- and the figure's name is the part a VLM could never supply.
    for clause in (entity, " and ".join(names)):
        if not clause:
            continue
        candidate = f"{TRIGGER}, {clause}, {body}"
        if count_tokens(candidate)[0] <= CLIP_CONTENT_BUDGET:
            return candidate, True
    return text, False


def cmd_import(args) -> int:
    rows = [json.loads(line) for line in
            args.source.read_text(encoding="utf-8").splitlines() if line.strip()]

    overrides: dict[str, dict] = {}
    if args.overrides and args.overrides.exists():
        for line in args.overrides.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                overrides[row["file_name"]] = row

    if _clip_tokeniser() is None:
        print("NOTE: transformers is not installed, so token counts are a "
              "conservative estimate\n      and some entity clauses may be "
              "skipped that would in fact fit.\n      pip install transformers "
              "for exact counts.\n", file=sys.stderr)

    store = sync_store(args.out, args.images)
    adopted = injected = applied = unknown = 0

    for row in rows:
        name = row["file_name"]
        slug = Path(name).stem
        entry = store.get(slug)
        if entry is None:
            print(f"  ! {name} is not a kept image in the manifest -- skipped", file=sys.stderr)
            unknown += 1
            continue

        override = overrides.get(name)
        text = (override or row)["text"]
        note = override["note"] if override else ""
        if override:
            applied += 1

        text, did = inject_entity(text, entry.entity)
        injected += did

        entry.draft = row["text"]          # keep the upstream text for provenance
        entry.caption = text
        entry.note = note
        # Imported captions are drafts, not reviewed work: §1.3 discipline is the
        # human pass, and nothing here has been through it.
        entry.reviewed = False
        adopted += 1

    save_captions(args.out, store)
    print(f"  adopted   {adopted}")
    print(f"  overrides applied {applied}")
    print(f"  entity clause injected {injected}")
    if unknown:
        print(f"  skipped (not in manifest) {unknown}")
    print(f"\nWritten to {captions_path(args.out)}")
    print("Next: python scripts/caption_corpus.py lint")
    return 0


# ---------------------------------------------------------------------------
# review -- local, no GPU
# ---------------------------------------------------------------------------

def open_image(path: Path) -> None:
    opener = {"linux": "xdg-open", "darwin": "open"}.get(sys.platform, None)
    if sys.platform.startswith("win"):
        os.startfile(path)                                  # type: ignore[attr-defined]
        return
    if opener is None:
        return
    try:
        subprocess.Popen([opener, str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def input_prefilled(prompt: str, text: str) -> str:
    """Hand the user an editable line rather than making them retype the draft."""
    try:
        import readline
    except ImportError:
        print(f"  (draft) {text}")
        return input(prompt)

    def hook():
        readline.insert_text(text)
        readline.redisplay()

    readline.set_startup_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def cmd_review(args) -> int:
    store = sync_store(args.out, args.images)
    items = select(store, args.shard, args.split)

    if args.accept_all:
        # For a correction pass carried out outside this tool. It still refuses
        # anything that fails the lint -- "we read them all" is not a reason to
        # let a style word through.
        accepted, refused = 0, []
        for entry in items:
            if not entry.caption:
                continue
            errors, _ = lint_caption(entry.caption)
            if errors:
                refused.append((entry, errors))
                continue
            entry.reviewed = True
            accepted += 1
        save_captions(args.out, store)
        print(f"marked {accepted} captions reviewed")
        for entry, errors in refused:
            print(f"  REFUSED {entry.file}: {'; '.join(errors)}", file=sys.stderr)
        if refused:
            print(f"\n{len(refused)} still fail the lint and were left unreviewed.", file=sys.stderr)
            return 1
        return 0

    queue = [c for c in items if args.force or not c.reviewed]
    done = len(items) - len(queue)

    if not queue:
        print(f"All {len(items)} captions in this shard are reviewed.")
        return 0

    print(f"{len(queue)} to review ({done} already done).")
    print("Edit the line and press Enter to accept. Commands: /skip /flag <note> "
          "/open /redraft /quit\n")

    for i, entry in enumerate(queue, 1):
        path = args.images / entry.file
        header = f"[{i}/{len(queue)}] {entry.file}"
        print("=" * len(header))
        print(header)
        if entry.title:
            print(f"  title    {entry.title}")
        if entry.entity:
            print(f"  entity   {entry.entity}")
        if entry.parts:
            for key, _ in QUESTIONS:
                if entry.parts.get(key):
                    print(f"  {key:8s} {entry.parts[key]}")
        if not args.no_open and path.exists():
            open_image(path)

        while True:
            text = input_prefilled("  > ", entry.caption or entry.draft).strip()
            command = text.lower()

            if command in ("/quit", "/q"):
                save_captions(args.out, store)
                print(f"\nStopped. Progress saved to {captions_path(args.out)}")
                return 0
            if command in ("/skip", "/s"):
                break
            if command == "/open":
                open_image(path)
                continue
            if command == "/redraft":
                entry.caption = entry.draft
                continue
            if command.startswith("/flag"):
                entry.note = text[5:].strip()
                entry.reviewed = True
                print(f"  flagged: {entry.note or '(no note)'}")
                break

            errors, warnings = lint_caption(text)
            for warning in warnings:
                print(f"  warn:  {warning}")
            if errors:
                for error in errors:
                    print(f"  ERROR: {error}")
                print("  Not saved. Fix it, or /flag <note> to move on.")
                entry.caption = text
                continue

            entry.caption = text
            entry.reviewed = True
            n, exact = count_tokens(text)
            print(f"  saved ({n} tokens{'' if exact else ' est'})")
            break

        save_captions(args.out, store)

    save_captions(args.out, store)
    print(f"\nShard complete. {captions_path(args.out)}")
    return 0


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------

def cmd_lint(args) -> int:
    store = sync_store(args.out, args.images)
    items = select(store, args.shard, args.split)
    if not items:
        sys.exit("no captions found -- run the draft step first")

    missing = [c for c in items if not c.caption]
    unreviewed = [c for c in items if c.caption and not c.reviewed]
    flagged = [c for c in items if c.note]
    bad: list[tuple[Caption, list[str]]] = []
    warned: list[tuple[Caption, list[str]]] = []
    lengths: list[int] = []
    exactness = True

    for entry in items:
        if not entry.caption:
            continue
        errors, warnings = lint_caption(entry.caption)
        n, exact = count_tokens(entry.caption)
        exactness &= exact
        lengths.append(n)
        if errors:
            bad.append((entry, errors))
        if warnings:
            warned.append((entry, warnings))

    print(f"captions      {len(items) - len(missing)}/{len(items)}")
    print(f"reviewed      {sum(1 for c in items if c.reviewed)}/{len(items)}")
    if lengths:
        label = "tokens" if exactness else "tokens (estimated)"
        print(f"{label:<13} min {min(lengths)}  mean {sum(lengths) // len(lengths)}  max {max(lengths)}")
    if not exactness:
        print("              (transformers not installed; counts are a padded estimate)")

    if missing:
        print(f"\nno caption yet ({len(missing)}):")
        for entry in missing[:15]:
            print(f"  {entry.file}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")

    if flagged:
        print(f"\nflagged for a second look ({len(flagged)}):")
        for entry in flagged:
            print(f"  {entry.file}: {entry.note}")

    if warned and args.verbose:
        print(f"\nwarnings ({len(warned)}):")
        for entry, warnings in warned:
            print(f"  {entry.file}: {'; '.join(warnings)}")

    if bad:
        print(f"\nERRORS ({len(bad)}):")
        for entry, errors in bad:
            print(f"  {entry.file}")
            for error in errors:
                print(f"      {error}")
            print(f"      {entry.caption}")

    if unreviewed:
        print(f"\n{len(unreviewed)} captions are still machine drafts. "
              "The human pass is what keeps style words out (plans/PLAN.md §1.3).")

    print()
    if bad:
        print("FAIL -- fix the errors above before training.")
        return 1
    if missing or unreviewed:
        # Not a failure: the corpus is mid-pass. But it is not a green light either,
        # and printing PASS here would be a false one.
        print("INCOMPLETE -- no errors, but the pass is not finished.")
        return 0
    print("PASS")
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def cmd_export(args) -> int:
    store = sync_store(args.out, args.images)
    everything = select(store, None, None)
    items = [c for c in everything if c.caption]
    if not items:
        sys.exit("nothing to export -- run the draft and review steps first")

    # Uncaptioned images drop out of `items` silently, so without this check a
    # quarter-finished corpus exports as a valid-looking 25-image dataset and the
    # shortfall only shows up as a bad LoRA days later.
    # Only the train split needs captions. The holdout is scored by image-embedding
    # distance in Milestone 4, so captioning it would be wasted effort.
    uncaptioned = [c for c in everything if not c.caption and c.split == "train"]
    if uncaptioned and not args.allow_incomplete:
        n_train = sum(1 for c in everything if c.split == "train")
        print(f"{len(uncaptioned)} of {n_train} train images have no caption:", file=sys.stderr)
        for entry in uncaptioned[:10]:
            print(f"  {entry.file}", file=sys.stderr)
        if len(uncaptioned) > 10:
            print(f"  ... and {len(uncaptioned) - 10} more", file=sys.stderr)
        sys.exit("\nRefusing to export a partial corpus. Pass --allow-incomplete to override.")

    blocking = [(c, e) for c in items for e in [lint_caption(c.caption)[0]] if e]
    if blocking and not args.allow_lint_failures:
        print(f"{len(blocking)} captions fail the lint:", file=sys.stderr)
        for entry, errors in blocking[:10]:
            print(f"  {entry.file}: {'; '.join(errors)}", file=sys.stderr)
        sys.exit("\nRefusing to export. Fix them, or pass --allow-lint-failures.")

    unreviewed = sum(1 for c in items if not c.reviewed and c.split == "train")
    if unreviewed and not args.allow_unreviewed:
        sys.exit(f"{unreviewed} captions are unreviewed machine drafts. "
                 "Run the review pass, or pass --allow-unreviewed.")

    # The holdout carries no captions, but Milestone 4 still needs it as a clean
    # directory of its own -- scoring it out of curated/oil means filtering the
    # manifest by split every time, and risks pointing training at it by mistake.
    holdout = [c for c in everything if c.split == "holdout"]
    if holdout:
        dest = args.out / "holdout"
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for entry in holdout:
            source = args.images / entry.file
            if source.exists() and not (dest / entry.file).exists():
                shutil.copy2(source, dest / entry.file)
                copied += 1
        print(f"  holdout  {len(holdout):3d} images -> {dest}  (no captions; "
              f"style-similarity reference)")

    written = {}
    for split in ("train",):
        rows = [c for c in items if c.split == split]
        if not rows:
            continue
        dest = args.out / split
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "metadata.jsonl").open("w", encoding="utf-8") as fh:
            for entry in rows:
                source = args.images / entry.file
                if not source.exists():
                    print(f"  ! missing image {source}", file=sys.stderr)
                    continue
                target = dest / entry.file
                if not target.exists():
                    shutil.copy2(source, target)
                # diffusers' imagefolder loader keys on file_name and reads "text".
                fh.write(json.dumps({"file_name": entry.file, "text": entry.caption},
                                    ensure_ascii=False) + "\n")
        written[split] = len(rows)
        print(f"  {split:8s} {len(rows):3d} images -> {dest}")

    print(f"\nExported {sum(written.values())} captioned images.")
    print("Train with:  --train_data_dir "
          f"{args.out / 'train'}  --caption_column text")
    if "holdout" in written:
        print(f"The holdout set is for style-similarity evaluation only -- never train on it.")
    return 0


# ---------------------------------------------------------------------------
# card -- document the caption pass
#
# Written as its own file rather than appended to DATA_CARD.md, because the
# scraper regenerates that one wholesale and would wipe anything added here.
# ---------------------------------------------------------------------------

def cmd_card(args) -> int:
    store = sync_store(args.out, args.images)
    items = select(store, None, None)
    captioned = [c for c in items if c.caption]
    train = [c for c in captioned if c.split == "train"]
    lengths = [count_tokens(c.caption)[0] for c in captioned]
    exact = all(count_tokens(c.caption)[1] for c in captioned)
    overrides = [c for c in captioned if c.note]
    # Count by whether the figure's name actually reached the caption. Comparing
    # against the draft misses the overridden records, five of which also carry
    # an injected clause.
    def _has_entity(c: Caption) -> bool:
        names = [n for n in c.entity.split(",")[0].split(" and ") if n[:1].isupper()]
        return bool(names) and any(n.lower() in c.caption.lower() for n in names)

    injected = [c for c in captioned if c.entity and _has_entity(c)]
    failing = [c for c in captioned if lint_caption(c.caption)[0]]

    src = args.source.name if args.source else "(unknown)"
    approx = "" if exact else " (estimated -- install transformers for exact counts)"

    lines = [
        "# Caption Card - Ravi Varma corpus",
        "",
        f"Generated by `scripts/caption_corpus.py card`. Captions live in "
        f"`data/captions.jsonl`; the training set is written to `data/train/`.",
        "",
        "## Counts",
        "",
        f"- Images captioned: **{len(captioned)}** (all of the {len(train)}-image train split)",
        f"- Holdout images: **{sum(1 for c in items if c.split == 'holdout')}**, "
        f"deliberately uncaptioned -- Milestone 4 scores them by image-embedding",
        "  distance, so captions would be wasted effort",
        f"- Reviewed by a human: **{sum(1 for c in captioned if c.reviewed)}/{len(captioned)}**",
        f"- Failing the style lint: **{len(failing)}**",
        "",
        "## Trigger token",
        "",
        f"Every caption begins with `{TRIGGER}`, a rare non-semantic token "
        f"(`plans/PLAN.md` §1.4).",
        "",
        "An earlier pass used `ravi_varma_style`. That was replaced: it costs 6 CLIP",
        "tokens against a 75-token budget where `rvrma` costs 3, and it contains the",
        "word *style*, which would be trained into a very common token during the",
        "planned text-encoder ablation.",
        "",
        f"The trigger appears in {len(captioned)}/{len(captioned)} captions, so there is no",
        "contrastive signal distinguishing \"trigger causes the style\" from \"the style is",
        "unconditional\". This is normal for a single-concept style LoRA -- the LoRA",
        "*weight*, not the token, is the real control -- but it means the trigger should",
        "not be assumed to work as a switch without testing it.",
        "",
        "## Captions describe content, never style",
        "",
        "`plans/PLAN.md` §1.3. A caption containing \"oil painting\" or \"soft lighting\"",
        "risks binding Varma's look to those common English tokens rather than to the",
        "trigger. Enforced mechanically, not by convention:",
        "",
        "- `review` refuses to save a caption that trips the lint",
        "- `export` refuses a corpus that is incomplete, unreviewed, or failing lint",
        "",
        "The lint distinguishes a banned word from the same word naming a real object:",
        "an oil *lamp* is a prop, an oil *painting* is the medium. `ornate` and",
        "`intricate` are treated as content (\"ornate jewelry\", \"intricate patterns\").",
        "",
        "## Token budget",
        "",
        f"SD 1.5's CLIP text encoder truncates at {CLIP_MAX_TOKENS} tokens, silently. "
        f"Measured{approx}:",
        "",
        "| | tokens |",
        "|---|---|",
        f"| min | {min(lengths)} |",
        f"| median | {sorted(lengths)[len(lengths) // 2]} |",
        f"| mean | {sum(lengths) // len(lengths)} |",
        f"| max | {max(lengths)} |",
        f"| over the {CLIP_CONTENT_BUDGET}-token budget | {sum(1 for n in lengths if n > CLIP_CONTENT_BUDGET)} |",
        "",
        "## Provenance",
        "",
        f"Content captions were produced by a vision-language model against a fixed",
        f"ten-field template (subject, pose, gaze, clothing, props, setting,",
        f"composition, lighting, palette, medium) and imported from `{src}`.",
        "",
        "The lighting and medium fields were dropped at source: they carried almost all",
        "the style language (57/60 and 32/60 of cleanly-parsed records respectively),",
        "while fields 1-7 and 9 were essentially clean. Composition and palette were",
        "kept -- composition serves Milestone 4's interest in Varma's balanced framing,",
        "and palette carries background colour the clothing field does not.",
        "",
        "### Entity clauses",
        "",
        f"**{len(injected)}** captions were given a ground-truth entity clause drawn from the",
        "file title and Commons categories, inserted after the trigger:",
        "",
        "```",
        "rvrma, Ahalya, Woman in traditional attire, Leaning against a tree branch, ...",
        "```",
        "",
        "A VLM cannot supply these -- it sees a woman leaning on a branch, not Ahalya.",
        "The Commons category tree alone is too thin (only 28 of 120 kept files carry an",
        "epic label, and *Ahalya by Ravi Varma* is filed under nothing more specific than",
        "\"women of India\"), so names are recovered from titles as well, with a",
        "false-friend list to stop *Lakshmipuram* becoming the goddess Lakshmi and",
        "*Princess Tarabai of Baroda* becoming Tara of the Ramayana.",
        "",
        "Where the full clause would breach the token budget the epic is dropped and the",
        "bare names kept, rather than losing the name or truncating the caption.",
        "",
    ]

    if overrides:
        lines += [
            "### Hand corrections",
            "",
            f"**{len(overrides)}** captions were corrected by hand in "
            f"`data/caption_overrides.jsonl`, applied on top of the imported text so the",
            "upstream file stays the source of record:",
            "",
            "| Image | Reason |",
            "|---|---|",
        ]
        lines += [f"| `{c.file}` | {c.note} |" for c in sorted(overrides, key=lambda c: c.file)]
        lines += [""]

    lines += [
        "## Known limitations",
        "",
        "- Captions were machine-generated and corrected, not written from scratch. The",
        "  human pass reviewed them; it did not independently re-describe every painting.",
        "- Field order is fixed, so every caption has the same shape. This is good for",
        "  consistency and bad for prompt diversity -- user prompts at inference will not",
        "  resemble this structure.",
        f"- {sum(1 for c in captioned if count_tokens(c.caption)[0] > CLIP_CONTENT_BUDGET - 8)} "
        f"captions sit within 8 tokens of the budget, so any later addition truncates.",
        "",
    ]

    dest = args.out / "CAPTION_CARD.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {dest}")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"), help="data directory (default: data)")
    ap.add_argument("--images", type=Path, default=Path("data/curated/oil"),
                    help="curated image directory (default: data/curated/oil)")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("draft", help="BLIP-2 draft captions (needs a GPU)")
    d.add_argument("--model", default="Salesforce/blip2-opt-2.7b")
    d.add_argument("--limit", type=int, help="stop after N images (smoke test)")
    d.add_argument("--shard", help="split the work, e.g. 1/3")
    d.add_argument("--force", action="store_true", help="redraft captions that already exist")
    d.set_defaults(func=cmd_draft)

    i = sub.add_parser("import", help="adopt captions produced by another tool")
    i.add_argument("--source", type=Path, default=Path("lora_metadata_final.jsonl"),
                   help="jsonl with file_name/text (default: lora_metadata_final.jsonl)")
    i.add_argument("--overrides", type=Path, default=Path("data/caption_overrides.jsonl"),
                   help="hand corrections applied on top of the source")
    i.set_defaults(func=cmd_import)

    r = sub.add_parser("review", help="human correction pass (local, no GPU)")
    r.add_argument("--shard", help="split the work between teammates, e.g. 1/3")
    r.add_argument("--split", choices=["train", "holdout"])
    r.add_argument("--force", action="store_true", help="revisit captions already reviewed")
    r.add_argument("--no-open", action="store_true", help="do not launch an image viewer")
    r.add_argument("--accept-all", action="store_true",
                   help="mark every lint-clean caption reviewed, for a pass done elsewhere")
    r.set_defaults(func=cmd_review)

    l = sub.add_parser("lint", help="validate every caption")
    l.add_argument("--shard")
    l.add_argument("--split", choices=["train", "holdout"])
    l.add_argument("--verbose", "-v", action="store_true", help="also list warnings")
    l.set_defaults(func=cmd_lint)

    c = sub.add_parser("card", help="write CAPTION_CARD.md documenting the pass")
    c.add_argument("--source", type=Path, default=Path("lora_metadata_final.jsonl"))
    c.set_defaults(func=cmd_card)

    e = sub.add_parser("export", help="write metadata.jsonl for diffusers")
    e.add_argument("--allow-incomplete", action="store_true",
                   help="export even though some images have no caption")
    e.add_argument("--allow-unreviewed", action="store_true")
    e.add_argument("--allow-lint-failures", action="store_true")
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
