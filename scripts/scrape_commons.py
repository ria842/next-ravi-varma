#!/usr/bin/env python3
"""
Build the Raja Ravi Varma training corpus from Wikimedia Commons.

Pipeline
--------
1. discover : deepcategory search over "Paintings by Raja Ravi Varma", then batched
              imageinfo + category lookups for every hit.
2. classify : separate original oils from Ravi Varma Press prints, and drop crops,
              book scans, photographs of paintings in situ, and low-resolution files.
3. download : fetch a width-capped rendering of each surviving file.
4. dedup    : perceptual-hash (dhash) near-duplicate detection; keep the largest copy.
5. split    : stratified train/holdout split for style-similarity evaluation.

Outputs data/raw/, data/curated/{oil,print}/, data/manifest.jsonl and data/DATA_CARD.md.

Ravi Varma died in 1906, so the originals are public domain; the Commons files carry
PD-Art / PD-old tags. Re-running is safe -- completed downloads are skipped.

Usage
-----
    python scripts/scrape_commons.py                 # full pipeline
    python scripts/scrape_commons.py --dry-run       # metadata only, no downloads
    python scripts/scrape_commons.py --max-width 4096
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Wikimedia's User-Agent policy requires a descriptive agent with contact info.
# Requests without one are refused with HTTP 403.
USER_AGENT = (
    "NextRaviVarma-AcademicProject/0.1 "
    "(https://github.com/ria842/next-ravi-varma; academic coursework)"
)
API = "https://commons.wikimedia.org/w/api.php"
ROOT_CATEGORY = "Paintings by Raja Ravi Varma"

REQUEST_DELAY = 0.3          # polite pause between API calls, seconds
BATCH_SIZE = 50              # titles per imageinfo query (API limit for anon users)
MIN_EDGE = 460               # short edge floor; 460 still supports 512px training
MIN_PIXELS = 460 * 600       # reject small files regardless of aspect
MAX_ASPECT = 2.6             # reject extreme strips (usually panels or banners)
DHASH_SIZE = 8               # dhash grid -> 64-bit signature
HAMMING_THRESHOLD = 6        # <= this distance counts as a near-duplicate
HOLDOUT_N = 20               # paintings reserved for evaluation
RANDOM_SEED = 20260921


# ---------------------------------------------------------------------------
# Heuristics
#
# Note: Commons categories do NOT reliably distinguish oleographs from oils.
# "Lakshmi Embellished, by Ravi Varma Press" is filed under "1890 paintings by
# Raja Ravi Varma" with no lithograph category at all. Title and description text
# are the only dependable signals, so the print detector works off those.
# ---------------------------------------------------------------------------

# Strong print signals. Checked against the title always, and against the description
# only for unambiguous terms -- museum credit text is full of incidental words.
PRINT_TITLE_PATTERNS = [
    r"ravi\s*varma\s*press",
    r"chromolithograph",
    r"oleograph",
    r"lithograph",
    r"\bposter\b",
    r"calendar\s*art",
]
PRINT_DESC_PATTERNS = [
    r"ravi\s*varma\s*press",
    r"chromolithograph",
    r"oleograph",
]
# Commons categories are incomplete for prints -- "Lakshmi Embellished, by Ravi Varma
# Press" is filed as a painting with no print category at all -- but where a print
# category IS present it is authoritative. Used as a third positive signal: it is the
# only thing that catches "Krishna as Envoy", whose title and description are silent.
PRINT_CATEGORY_PATTERNS = [r"lithograph", r"oleograph", r"\bprints?\b"]

# Files that are not a clean, whole, primary reproduction of a painting.
#
# These are matched against the TITLE ONLY. Matching descriptions here was tried and
# produced bad false positives: "Reclining Woman" (4024x2840) and "Rani Bharani
# Thirunal" (3000x3790) were both dropped as book scans because their museum credit
# lines happen to mention a book. Commons titles are curated and reliable; the
# description field is free-text boilerplate and is not safe to filter on.
TITLE_REJECT_PATTERNS = [
    (r"\(detail\)|\bdetail\b", "detail_crop"),
    (r"\bcropped?\b|\(page\s*\d+", "page_or_crop"),
    (r",\s*edit\b|\bedit\.(jpg|png)", "edited_variant"),
    (r"\bsignature\b", "signature"),
    (r"\bstamp\b|postage|colnect", "stamp"),
    (r"\bstatue\b|\bsculpture\b|\bbust\b", "not_a_painting"),
    # "Painting in <somewhere>" is a snapshot of a framed work on a wall. Deliberately
    # narrow: a bare "in <x> palace" also matches portraits whose title merely records
    # the holding collection (e.g. "... Changanassery Lakshmipuram Palace", a genuine
    # 4164x5334 painting), which must stay in the corpus.
    (r"\bpainting\s+in\b", "photo_in_situ"),
    (r"\batelier\b|\breplica\b", "reproduction"),
]

# Category patterns that yield a ground-truth subject label. BLIP-2 cannot produce
# these -- it will never guess "Shakuntala" -- so they are the backbone of captioning.
CATEGORY_PREFIXES = [
    "Depiction of scenes from the ",
    "Depiction of Hindu Gods and Goddesses",
    "Depiction of ",
    "Paintings of ",
    "Landscape paintings",
    "Musicians",
]
CATEGORY_SUFFIX = " by Raja Ravi Varma"
YEAR_CATEGORY = re.compile(r"^(\d{4})s?\s+paintings?\s+by\s+Raja\s+Ravi\s+Varma$", re.I)

EPICS = {
    "mahabharata": "Mahabharata",
    "ramayana": "Ramayana",
    "abhij": "Abhijnanashakuntalam",
    "shakuntal": "Abhijnanashakuntalam",
    "sakuntal": "Abhijnanashakuntalam",
}


@dataclass
class Record:
    """One Commons file and everything derived from it."""
    title: str
    pageid: int = 0
    width: int = 0
    height: int = 0
    mime: str = ""
    source_url: str = ""
    fetch_url: str = ""
    categories: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    epic: str | None = None
    year: str | None = None
    description: str = ""
    kind: str = "oil"           # oil | print
    status: str = "candidate"   # candidate | kept | rejected
    reason: str = ""
    local_path: str = ""
    dhash: str = ""
    split: str = ""

    @property
    def slug(self) -> str:
        base = re.sub(r"^File:", "", self.title)
        base = re.sub(r"\.(jpg|jpeg|png|tif|tiff)$", "", base, flags=re.I)
        base = re.sub(r"[^\w\s-]", "", base).strip()
        return re.sub(r"[\s_-]+", "_", base)[:110]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def api_get(session: requests.Session, params: dict) -> dict:
    """GET against the Commons API with retry on transient failure."""
    params = {**params, "format": "json", "formatversion": "2"}
    for attempt in range(4):
        try:
            r = session.get(API, params=params, timeout=60)
            if r.status_code == 403:
                sys.exit(
                    "HTTP 403 from Commons. This almost always means the User-Agent "
                    "was rejected -- check the USER_AGENT constant."
                )
            r.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == 3:
                raise
            wait = 2 ** attempt
            print(f"  ! request failed ({exc}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return {}


def discover_titles(session: requests.Session) -> list[str]:
    """Deep-category search returns files nested anywhere under the root category."""
    titles, offset = [], 0
    while True:
        data = api_get(session, {
            "action": "query",
            "list": "search",
            "srsearch": f'deepcategory:"{ROOT_CATEGORY}"',
            "srnamespace": 6,
            "srlimit": 200,
            "sroffset": offset,
        })
        hits = data.get("query", {}).get("search", [])
        titles.extend(h["title"] for h in hits)
        cont = data.get("continue", {}).get("sroffset")
        if cont is None:
            break
        offset = cont
    return titles


def fetch_metadata(session: requests.Session, titles: list[str], max_width: int) -> list[Record]:
    """Batched imageinfo + category lookup. Categories may paginate for busy files."""
    records: dict[str, Record] = {}
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i:i + BATCH_SIZE]
        print(f"  metadata {i + 1}-{i + len(batch)} of {len(titles)}")
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo|categories",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": max_width,
            "cllimit": "max",
            "clshow": "!hidden",
        }
        while True:
            data = api_get(session, params)
            for page in data.get("query", {}).get("pages", []):
                title = page.get("title", "")
                rec = records.get(title)
                if rec is None:
                    rec = Record(title=title, pageid=page.get("pageid", 0))
                    records[title] = rec
                    info = (page.get("imageinfo") or [{}])[0]
                    rec.width = info.get("width", 0)
                    rec.height = info.get("height", 0)
                    rec.mime = info.get("mime", "")
                    rec.source_url = info.get("url", "")
                    # thumburl is a width-capped rendering; far cheaper than an
                    # 8000px original we would immediately downscale anyway.
                    rec.fetch_url = info.get("thumburl") or info.get("url", "")
                    meta = info.get("extmetadata", {})
                    desc = meta.get("ImageDescription", {}).get("value", "")
                    rec.description = re.sub(r"<[^>]+>", " ", desc)[:600].strip()
                    rec.year = _year_from_meta(meta)
                for cat in page.get("categories", []):
                    rec.categories.append(cat["title"].replace("Category:", ""))
            cont = data.get("continue")
            if not cont:
                break
            params = {**params, **cont}
    return list(records.values())


def _year_from_meta(meta: dict) -> str | None:
    raw = meta.get("DateTimeOriginal", {}).get("value", "") or meta.get("DateTime", {}).get("value", "")
    m = re.search(r"(1[89]\d{2})", re.sub(r"<[^>]+>", " ", raw))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def derive_labels(rec: Record) -> None:
    """Pull subject, epic and year out of the category tree."""
    subjects: list[str] = []
    for cat in rec.categories:
        ym = YEAR_CATEGORY.match(cat)
        if ym:
            rec.year = rec.year or ym.group(1)
            continue
        if not cat.endswith(CATEGORY_SUFFIX):
            continue
        label = cat[: -len(CATEGORY_SUFFIX)].strip()
        if re.match(r"^\d", label) or label.lower().startswith("paintings by"):
            continue
        for prefix in CATEGORY_PREFIXES:
            if label.startswith(prefix):
                label = label[len(prefix):].strip() or prefix.strip()
                break
        if any(re.search(p, label.lower()) for p in PRINT_CATEGORY_PATTERNS):
            continue
        if label and label.lower() not in {"paintings", "works"}:
            subjects.append(label)

    rec.subjects = sorted(set(subjects))
    haystack = " ".join(rec.categories + [rec.title]).lower()
    for needle, epic in EPICS.items():
        if needle in haystack:
            rec.epic = epic
            break


def classify(rec: Record, min_edge: int = MIN_EDGE) -> None:
    """Assign kind (oil vs print) and accept/reject with a recorded reason."""
    title = rec.title.lower()
    desc = rec.description.lower()

    cats = " | ".join(rec.categories).lower()
    if any(re.search(p, title) for p in PRINT_TITLE_PATTERNS) or \
       any(re.search(p, desc) for p in PRINT_DESC_PATTERNS) or \
       any(re.search(p, cats) for p in PRINT_CATEGORY_PATTERNS):
        rec.kind = "print"

    if not rec.mime.startswith("image/"):
        rec.status, rec.reason = "rejected", "not_an_image"
        return

    # Prints are excluded from the LoRA corpus but still downloaded, so report them
    # under their own reason rather than letting a generic rule claim them first.
    if rec.kind == "print":
        rec.status, rec.reason = "rejected", "ravi_varma_press_print"
        return

    for pattern, reason in TITLE_REJECT_PATTERNS:
        if re.search(pattern, title):
            rec.status, rec.reason = "rejected", reason
            return

    short, long = min(rec.width, rec.height), max(rec.width, rec.height)
    if short < min_edge or rec.width * rec.height < MIN_PIXELS:
        rec.status, rec.reason = "rejected", "low_resolution"
        return
    if short and long / short > MAX_ASPECT:
        rec.status, rec.reason = "rejected", "extreme_aspect_ratio"
        return

    rec.status = "kept"


# ---------------------------------------------------------------------------
# Download and perceptual dedup
# ---------------------------------------------------------------------------

def download(session: requests.Session, records: list[Record], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = [r for r in records if r.status == "kept" or r.kind == "print"]
    for i, rec in enumerate(targets, 1):
        ext = ".png" if rec.mime.endswith("png") else ".jpg"
        dest = raw_dir / f"{rec.slug}{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            rec.local_path = str(dest)
            continue
        try:
            r = session.get(rec.fetch_url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            rec.local_path = str(dest)
            print(f"  [{i}/{len(targets)}] {dest.name}")
            time.sleep(REQUEST_DELAY)
        except requests.RequestException as exc:
            rec.status, rec.reason = "rejected", "download_failed"
            print(f"  ! {rec.title}: {exc}", file=sys.stderr)


def dhash(path: str, size: int = DHASH_SIZE) -> str | None:
    """Difference hash: compare each pixel to its right neighbour on a tiny grey grid.

    Robust to rescaling and mild colour shifts, which is exactly how Commons
    near-duplicates differ (same painting, different scan or crop).
    """
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            px = np.asarray(im, dtype=np.int16)
    except Exception as exc:                      # unreadable or truncated file
        print(f"  ! hash failed for {path}: {exc}", file=sys.stderr)
        return None
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def dedup(records: list[Record]) -> int:
    """Greedy near-duplicate removal; the highest-resolution copy of each group wins."""
    kept = [r for r in records if r.status == "kept" and r.local_path]
    for rec in kept:
        rec.dhash = dhash(rec.local_path) or ""

    hashed = [r for r in kept if r.dhash]
    hashed.sort(key=lambda r: r.width * r.height, reverse=True)

    survivors: list[Record] = []
    removed = 0
    for rec in hashed:
        value = int(rec.dhash, 16)
        clash = next(
            (s for s in survivors if bin(value ^ int(s.dhash, 16)).count("1") <= HAMMING_THRESHOLD),
            None,
        )
        if clash is not None:
            rec.status = "rejected"
            rec.reason = f"near_duplicate_of:{clash.slug}"
            removed += 1
        else:
            survivors.append(rec)
    return removed


def assign_splits(records: list[Record]) -> None:
    """Stratify the holdout across subjects so evaluation is not all portraits."""
    kept = [r for r in records if r.status == "kept"]
    rng = np.random.default_rng(RANDOM_SEED)

    buckets: dict[str, list[Record]] = defaultdict(list)
    for rec in kept:
        buckets[rec.epic or (rec.subjects[0] if rec.subjects else "unlabelled")].append(rec)

    for rec in kept:
        rec.split = "train"

    holdout: list[Record] = []
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(holdout) < min(HOLDOUT_N, len(kept)):
        progressed = False
        for key in order:
            pool = [r for r in buckets[key] if r.split == "train"]
            if not pool:
                continue
            pick = pool[int(rng.integers(len(pool)))]
            pick.split = "holdout"
            holdout.append(pick)
            progressed = True
            if len(holdout) >= min(HOLDOUT_N, len(kept)):
                break
        if not progressed:
            break


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def organise(records: list[Record], curated: Path) -> None:
    import shutil
    for kind in ("oil", "print"):
        (curated / kind).mkdir(parents=True, exist_ok=True)
    for rec in records:
        if not rec.local_path or not Path(rec.local_path).exists():
            continue
        if rec.status == "kept":
            dest = curated / "oil" / Path(rec.local_path).name
        elif rec.kind == "print":
            dest = curated / "print" / Path(rec.local_path).name
        else:
            continue
        if not dest.exists():
            shutil.copy2(rec.local_path, dest)


def write_outputs(records: list[Record], data_dir: Path) -> Counter:
    with (data_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: r.title):
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    reasons = Counter(r.reason.split(":")[0] for r in records if r.status == "rejected")
    kept = [r for r in records if r.status == "kept"]
    subjects = Counter(s for r in kept for s in r.subjects)
    epics = Counter(r.epic for r in kept if r.epic)

    lines = [
        "# Data Card - Ravi Varma corpus",
        "",
        f"Generated by `scripts/scrape_commons.py` from Wikimedia Commons "
        f'category tree `{ROOT_CATEGORY}`.',
        "",
        "## Counts",
        "",
        f"- Files discovered: **{len(records)}**",
        f"- Kept for training: **{len(kept)}** "
        f"(train {sum(1 for r in kept if r.split == 'train')}, "
        f"holdout {sum(1 for r in kept if r.split == 'holdout')})",
        f"- Ravi Varma Press prints, retained separately: "
        f"**{sum(1 for r in records if r.kind == 'print')}**",
        "",
        "## Rejections by reason",
        "",
        "| Reason | Count |",
        "|---|---|",
    ]
    lines += [f"| {reason} | {n} |" for reason, n in reasons.most_common()]
    lines += [
        "",
        "## Subject coverage (kept files)",
        "",
        "| Subject | Count |",
        "|---|---|",
    ]
    lines += [f"| {s} | {n} |" for s, n in subjects.most_common(30)]
    lines += [
        "",
        "## Epic coverage",
        "",
        "| Epic | Count |",
        "|---|---|",
    ]
    lines += [f"| {e} | {n} |" for e, n in epics.most_common()]
    lines += [
        "",
        "## Curation decisions",
        "",
        "- **Ravi Varma Press prints are excluded from training.** The chromolithographs",
        "  are plentiful but look flat and printed; mixing them in blurs the oil-painting",
        "  aesthetic the LoRA is meant to learn. They are kept under `curated/print/` for",
        "  a side-by-side figure in the report.",
        "- Commons categories do **not** reliably mark prints -- \"Lakshmi Embellished, by",
        "  Ravi Varma Press\" is filed as a painting -- so detection uses title and",
        "  description text instead.",
        "- Details, crops, book scans, edited variants and photographs of paintings hung",
        "  in situ are rejected: they are not clean reproductions of a whole work.",
        f"- Near-duplicates are removed by 64-bit dhash at Hamming distance "
        f"<= {HAMMING_THRESHOLD}; the highest-resolution copy of each group survives.",
        f"- The {HOLDOUT_N}-image holdout is stratified by epic/subject and is never seen",
        "  in training. It is the reference set for style-similarity evaluation.",
        "",
        "## Licensing",
        "",
        "Raja Ravi Varma died in 1906; the original works are public domain and the",
        "Commons files carry PD-Art / PD-old tags. Reproduction photographs may carry",
        "their own credits -- `manifest.jsonl` records the source URL for every file.",
        "",
    ]
    (data_dir / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    return reasons


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    ap.add_argument("--max-width", type=int, default=2048,
                    help="cap downloaded width; 2048 is ample for 512-768px training")
    ap.add_argument("--min-edge", type=int, default=MIN_EDGE,
                    help=f"short-edge floor in pixels (default {MIN_EDGE}); "
                         "raise to 700 only if training at 768px")
    ap.add_argument("--dry-run", action="store_true", help="metadata and classification only")
    args = ap.parse_args()

    data_dir: Path = args.out
    data_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()

    print(f"[1/5] discovering files under '{ROOT_CATEGORY}'")
    titles = discover_titles(session)
    print(f"      {len(titles)} files found")

    print("[2/5] fetching metadata")
    records = fetch_metadata(session, titles, args.max_width)

    print("[3/5] classifying")
    for rec in records:
        derive_labels(rec)
        classify(rec, args.min_edge)
    print(f"      {sum(1 for r in records if r.status == 'kept')} candidates, "
          f"{sum(1 for r in records if r.kind == 'print')} prints")

    if args.dry_run:
        write_outputs(records, data_dir)
        print(f"\nDry run complete. See {data_dir / 'DATA_CARD.md'}")
        return 0

    print("[4/5] downloading")
    download(session, records, data_dir / "raw")

    print("[5/5] deduplicating and splitting")
    removed = dedup(records)
    print(f"      {removed} near-duplicates removed")
    assign_splits(records)
    organise(records, data_dir / "curated")

    write_outputs(records, data_dir)
    kept = sum(1 for r in records if r.status == "kept")
    print(f"\nCorpus ready: {kept} images in {data_dir / 'curated' / 'oil'}")
    print(f"Manifest: {data_dir / 'manifest.jsonl'}")
    print(f"Data card: {data_dir / 'DATA_CARD.md'}")
    if kept < 90:
        print(f"\nWARNING: only {kept} images survived. The plan assumes 110-130.\n"
              "Consider relaxing MIN_EDGE or supplementing from WikiArt.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
