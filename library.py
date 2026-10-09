"""Library management on top of the image folder: favorites, trash/move/rename with undo,
auto albums from stored descriptions, and duplicate/similar finder.

Nothing here talks to the network. Functions take `root` (the image root) explicitly so they can be
tested on a temp folder; the server wires them to HTTP.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
STATE_DIR = ".yunaviewer"
LOCK = threading.Lock()  # guards favorites.json and ops.json
FILE_OP_LOCK = threading.RLock()  # serializes multi-file operations and their rollback/journal


def state_dir(root: Path) -> Path:
    path = root / STATE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _write_json(path: Path, data) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def is_state_path(root: Path, path: Path) -> bool:
    return (root / STATE_DIR) in (path, *path.parents)


# --- quick content key: size + first/last 64 KB. Cheap enough to hash a whole folder on every listing ---

_quick_cache: dict[tuple[str, int, int], str] = {}


def quick_hash(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _quick_cache:
        if len(_quick_cache) > 50_000:  # entries are tiny, but a long-running server should not grow forever
            _quick_cache.clear()
        digest = hashlib.sha1(str(stat.st_size).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(65536))
            if stat.st_size > 131072:
                handle.seek(-65536, 2)
                digest.update(handle.read(65536))
        _quick_cache[key] = digest.hexdigest()
    return _quick_cache[key]


# --- favorites: stored by content key so they survive moves and renames ---

def load_favorites(root: Path) -> set[str]:
    with LOCK:
        return set(_read_json(root / STATE_DIR / "favorites.json", []))


def set_favorites(root: Path, paths: list[Path], value: bool) -> int:
    keys = {quick_hash(path) for path in paths}
    with LOCK:
        favorites = set(_read_json(root / STATE_DIR / "favorites.json", []))
        favorites = favorites | keys if value else favorites - keys
        _write_json(state_dir(root) / "favorites.json", sorted(favorites))
    return len(keys)


# --- star ratings (1-5) and short notes, keyed like favorites so they follow moved/renamed files ---

MAX_NOTE = 200


def load_marks(root: Path) -> dict[str, dict]:
    with LOCK:
        return _read_json(root / STATE_DIR / "marks.json", {})


def set_marks(root: Path, paths: list[Path], rating: int | None = None, note: str | None = None) -> int:
    """rating 0 clears it; note "" clears it; None leaves that field alone."""
    if rating is not None and not 0 <= rating <= 5:
        raise ValueError("별점은 0~5 사이여야 합니다.")
    keys = {quick_hash(path) for path in paths}
    with LOCK:
        marks = _read_json(root / STATE_DIR / "marks.json", {})
        for key in keys:
            mark = dict(marks.get(key, {}))
            if rating is not None:
                mark["rating"] = rating
            if note is not None:
                mark["note"] = note.strip()[:MAX_NOTE]
            mark = {k: v for k, v in mark.items() if v}
            if mark:
                marks[key] = mark
            else:
                marks.pop(key, None)
        _write_json(state_dir(root) / "marks.json", marks)
    return len(keys)


# --- undoable file operations (trash / move / rename), logged in ops.json ---

def _unique(directory: Path, name: str) -> Path:
    target = directory / name
    n = 2
    while target.exists():
        target = directory / f"{Path(name).stem}_{n}{Path(name).suffix}"
        n += 1
    return target


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _record(root: Path, kind: str, items: list[dict]) -> dict:
    op = {"id": uuid.uuid4().hex, "kind": kind, "time": time.time(), "items": items}
    with LOCK:
        ops = _read_json(root / STATE_DIR / "ops.json", [])
        # drop logged items whose moved file is gone (e.g. trash cleaned by hand) so the log stays small
        ops = [o for o in ops if any((root / i["to"]).exists() for i in o["items"])]
        ops.append(op)
        _write_json(state_dir(root) / "ops.json", ops)
    return op


def _check_movable(root: Path, path: Path) -> None:
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS or is_state_path(root, path):
        raise ValueError(f"옮길 수 없는 항목입니다: {path.name}")


def trash_files(root: Path, paths: list[Path]) -> dict:
    with FILE_OP_LOCK:
        if len(set(paths)) != len(paths):
            raise ValueError("같은 파일이 중복 선택되었습니다.")
        for path in paths:
            _check_movable(root, path)
        trash = state_dir(root) / "trash"
        trash.mkdir(exist_ok=True)
        moved: list[tuple[Path, Path]] = []
        try:
            for path in paths:
                target = trash / f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}__{path.name}"
                shutil.move(str(path), target)
                moved.append((target, path))
            items = [{"from": _rel(root, source), "to": _rel(root, target)} for target, source in moved]
            return _record(root, "trash", items)
        except Exception:
            for target, source in reversed(moved):
                if target.exists():
                    shutil.move(str(target), source)
            raise


def move_files(root: Path, paths: list[Path], dest: Path) -> dict:
    with FILE_OP_LOCK:
        if not dest.is_dir() or is_state_path(root, dest):
            raise ValueError("이동할 폴더가 올바르지 않습니다.")
        if len(set(paths)) != len(paths):
            raise ValueError("같은 파일이 중복 선택되었습니다.")
        for path in paths:
            _check_movable(root, path)
        sources = [path for path in paths if path.parent != dest]
        if not sources:
            raise ValueError("이미 그 폴더에 있습니다.")
        moved: list[tuple[Path, Path]] = []
        try:
            for path in sources:
                target = _unique(dest, path.name)
                shutil.move(str(path), target)
                moved.append((target, path))
            items = [{"from": _rel(root, source), "to": _rel(root, target)} for target, source in moved]
            return _record(root, "move", items)
        except Exception:
            for target, source in reversed(moved):
                if target.exists():
                    shutil.move(str(target), source)
            raise


def undo_op(root: Path, op_id: str) -> dict:
    """Move every item of a logged operation back. Existing files are never overwritten."""
    with FILE_OP_LOCK:
        with LOCK:
            ops = _read_json(root / STATE_DIR / "ops.json", [])
        op = next((o for o in ops if o["id"] == op_id), None)
        if op is None:
            raise ValueError("되돌릴 작업을 찾지 못했습니다.")
        restored = skipped = renamed = 0
        for item in reversed(op["items"]):
            source, origin = root / item["to"], root / item["from"]
            if not source.is_file():
                skipped += 1
                continue
            origin.parent.mkdir(parents=True, exist_ok=True)
            target = _unique(origin.parent, origin.name)
            shutil.move(str(source), target)
            restored += 1
            renamed += target != origin
        with LOCK:
            ops = [o for o in _read_json(root / STATE_DIR / "ops.json", []) if o["id"] != op_id]
            _write_json(state_dir(root) / "ops.json", ops)
        return {"restored": restored, "skipped": skipped, "renamed": renamed}


def restore_from_trash(root: Path, paths: list[Path]) -> int:
    """Put trashed files back where they came from (falls back to <root>/restored)."""
    with FILE_OP_LOCK:
        trash = root / STATE_DIR / "trash"
        if len(set(paths)) != len(paths):
            raise ValueError("같은 파일이 중복 선택되었습니다.")
        for path in paths:
            if path.parent != trash or not path.is_file():
                raise ValueError(f"휴지통 파일이 아닙니다: {path.name}")
        with LOCK:
            ops = _read_json(root / STATE_DIR / "ops.json", [])
        origin_of = {i["to"]: i["from"] for o in ops if o["kind"] == "trash" for i in o["items"]}
        moved: list[tuple[Path, Path]] = []
        try:
            for path in paths:
                origin = root / origin_of.get(_rel(root, path), f"restored/{path.name.split('__', 1)[-1]}")
                origin.parent.mkdir(parents=True, exist_ok=True)
                target = _unique(origin.parent, origin.name)
                shutil.move(str(path), target)
                moved.append((target, path))
            return len(moved)
        except Exception:
            for target, source in reversed(moved):
                if target.exists():
                    shutil.move(str(target), source)
            raise


def empty_trash(root: Path, paths: list[Path] | None = None) -> int:
    """Delete trashed files for good (all of them, or only `paths`, which must be in the trash) and drop their
    undo/restore entries. Returns how many went."""
    with FILE_OP_LOCK:
        trash = root / STATE_DIR / "trash"
        if paths is None:
            files = [p for p in trash.iterdir() if p.is_file()] if trash.is_dir() else []
        else:
            if len(set(paths)) != len(paths):
                raise ValueError("같은 파일이 중복 선택되었습니다.")
            for path in paths:
                if path.parent != trash or not path.is_file():
                    raise ValueError(f"휴지통 파일이 아닙니다: {path.name}")
            files = paths
        for path in files:
            path.unlink()
        with LOCK:
            ops = [o for o in _read_json(root / STATE_DIR / "ops.json", [])
                   if any((root / i["to"]).exists() for i in o["items"])]
            _write_json(state_dir(root) / "ops.json", ops)
        return len(files)


# --- generation info embedded by AI image tools (ComfyUI graph, A1111 parameters, AIGC labels) ---

PROMPT_INPUTS = ("prompt", "text", "model.prompt", "positive", "text_g")
MODEL_INPUTS = ("model", "ckpt_name", "unet_name", "lora_name")
SETTING_INPUTS = ("steps", "cfg", "sampler_name", "scheduler", "denoise", "resolution", "model.resolution",
                  "aspect_ratio", "model.aspect_ratio", "width", "height")


def comfy_info(graph: dict) -> dict:
    """Readable summary of a ComfyUI API graph (the "prompt" PNG chunk): which nodes made the image, with what
    prompt, models, seed and settings. Negative prompts are the text nodes wired into a sampler's "negative"."""
    nodes = {key: node for key, node in graph.items() if isinstance(node, dict)}
    negative = {str(value[0]) for node in nodes.values()
                for name, value in (node.get("inputs") or {}).items()
                if name == "negative" and isinstance(value, list) and value}
    info = {"tool": "ComfyUI", "generators": [], "prompts": [], "negative": [], "models": [], "seed": None,
            "settings": {}, "input_images": 0}
    for key, node in sorted(nodes.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        kind, inputs = node.get("class_type", ""), node.get("inputs") or {}
        title = (node.get("_meta") or {}).get("title") or kind
        if kind == "LoadImage":
            info["input_images"] += 1
            continue
        texts = [inputs[name] for name in PROMPT_INPUTS if isinstance(inputs.get(name), str) and inputs[name].strip()]
        texts = [t for t in texts if len(t) > 3 and not t.lower().endswith((".safetensors", ".ckpt", ".pth", ".gguf"))]
        if texts:
            (info["negative"] if key in negative else info["prompts"]).extend(texts)
            if key not in negative and not kind.startswith(("CLIPTextEncode", "TextEncode")):
                info["generators"].append(title)  # API nodes (Nano Banana, Seedream, Grok...) prompt themselves
        for name in MODEL_INPUTS:
            value = inputs.get(name)
            if isinstance(value, str) and value and value not in info["models"]:
                info["models"].append(value)
        for name in ("seed", "noise_seed"):
            if isinstance(inputs.get(name), int) and info["seed"] is None:
                info["seed"] = inputs[name]
        for name in SETTING_INPUTS:
            value = inputs.get(name)
            if isinstance(value, (int, float, str)) and not isinstance(value, bool) and value != "":
                info["settings"].setdefault(name.removeprefix("model."), value)
        if kind == "KSampler" or kind.startswith("KSampler"):
            info["generators"].append(title)
    info["generators"] = list(dict.fromkeys(info["generators"]))
    return info


def generation_info(image) -> dict:
    """{} when the file carries nothing. `image` is an open PIL image (its .info holds the PNG text chunks)."""
    meta = image.info
    if isinstance(meta.get("prompt"), str):
        try:
            return comfy_info(json.loads(meta["prompt"]))
        except (ValueError, TypeError, AttributeError):
            pass
    if isinstance(meta.get("parameters"), str):  # AUTOMATIC1111 / Forge: prompt, "Negative prompt:", settings line
        text = meta["parameters"]
        head, _, settings = text.rpartition("\nSteps:")
        prompt, _, negative = (head or text).partition("\nNegative prompt:")
        return {"tool": "Stable Diffusion WebUI", "prompts": [prompt.strip()],
                "negative": [negative.strip()] if negative.strip() else [],
                "settings": {"parameters": ("Steps:" + settings).strip()} if settings else {}}
    xmp = meta.get("XML:com.adobe.xmp") or meta.get("xmp") or ""
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", "ignore")
    if "AIGC" in xmp:  # China's TC260 label that AI services must embed; no prompt inside
        producer = re.search(r'"ContentProducer"\s*:\s*"([^"]*)"', xmp)
        return {"tool": "AI 생성 표시 (TC260 AIGC)", "producer": producer.group(1) if producer else ""}
    return {}


def list_dirs(root: Path) -> list[str]:
    """All folders under root (relative, sorted), without hidden ones. A symlinked folder directly under root
    is a mounted library and is entered; symlinks deeper down are not followed (no cycles)."""
    result = [""]
    for top in sorted(root.iterdir()):
        if top.name.startswith(".") or not top.is_dir():
            continue
        result.append(top.name)
        for current, names, _ in os.walk(top):  # starts inside a symlinked top, never follows inner links
            names[:] = sorted(n for n in names if not n.startswith(".") and not (Path(current) / n).is_symlink())
            result.extend(_rel(root, Path(current) / n) for n in names)
    return sorted(result)


def make_dir(root: Path, parent: Path, name: str) -> Path:
    name = name.strip()
    if not name or re.search(r'[\\/:*?"<>|]', name) or name.startswith("."):
        raise ValueError("폴더 이름이 올바르지 않습니다.")
    target = parent / name
    target.mkdir(exist_ok=True)
    return target


# --- auto albums from stored descriptions (no AI calls) ---

_EMPTY = {"none", "not visible", ""}
_STOP = {"a", "an", "and", "with", "the", "of", "in", "on", "worn", "over", "style"}


def _tokens(text: str) -> set[str]:
    text = (text or "").strip().lower()
    if text in _EMPTY or text.startswith("not visible"):
        return set()
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)) - _STOP


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


# colour families: cream/ivory/off-white count as white, tan as beige, and so on
COLOR_FAMILIES = {
    "white": "white", "cream": "white", "ivory": "white", "off-white": "white", "offwhite": "white",
    "beige": "beige", "tan": "beige", "camel": "beige", "khaki": "beige", "nude": "beige",
    "black": "black", "gray": "gray", "grey": "gray", "charcoal": "gray", "silver": "gray",
    "red": "red", "burgundy": "red", "maroon": "red", "crimson": "red", "pink": "pink", "blush": "pink",
    "orange": "orange", "rust": "orange", "yellow": "yellow", "gold": "yellow", "golden": "yellow",
    "mustard": "yellow", "green": "green", "olive": "green", "mint": "green", "sage": "green",
    "blue": "blue", "navy": "blue", "denim": "blue", "teal": "teal", "purple": "purple",
    "lavender": "purple", "violet": "purple", "brown": "brown",
}
# garment nouns by family; shirt/blouse/top are one family because the describer mixes them
GARMENT_FAMILIES = {
    "dress": "dress", "gown": "dress", "romper": "romper", "jumpsuit": "romper",
    "shirt": "shirt", "blouse": "shirt", "top": "shirt", "camisole": "shirt", "tank": "shirt", "bustier": "shirt",
    "jersey": "tee", "tee": "tee", "t-shirt": "tee", "tshirt": "tee",
    "sweater": "knit", "pullover": "knit", "cardigan": "cardigan", "hoodie": "hoodie", "jacket": "jacket",
    "coat": "jacket", "blazer": "jacket", "skirt": "skirt", "shorts": "shorts", "jeans": "pants",
    "pants": "pants", "trousers": "pants", "leggings": "pants", "bikini": "swim", "lingerie": "swim",
    "bra": "swim", "bralette": "swim",
}


STEMS = {"knitted": "knit", "ribbed": "rib", "ruffled": "ruffle", "pleated": "pleat", "printed": "print",
         "striped": "stripe", "embroidered": "embroider", "cropped": "crop"}


def _families(tokens: set[str], table: dict[str, str]) -> set[str]:
    return {table[t] for t in tokens if t in table}


def _normalize(tokens: set[str]) -> set[str]:
    """Colour words become their family (cream -> white), garment nouns their family (blouse -> shirt)."""
    return {("c:" + COLOR_FAMILIES[t]) if t in COLOR_FAMILIES else ("g:" + GARMENT_FAMILIES[t])
            if t in GARMENT_FAMILIES else STEMS.get(t, t) for t in tokens}


def _similar_garment(a: set[str], b: set[str]) -> bool:
    """Same garment despite different wording: high word overlap, no conflicting colour family and no
    conflicting garment type ("white oversized jersey with red graphics" vs "white branded jersey top")."""
    if not a or not b:
        return False
    colors_a, colors_b = _families(a, COLOR_FAMILIES), _families(b, COLOR_FAMILIES)
    if colors_a and colors_b and not colors_a & colors_b:
        return False  # another colour is another garment, however alike the wording
    nouns_a, nouns_b = _families(a, GARMENT_FAMILIES), _families(b, GARMENT_FAMILIES)
    if nouns_a and nouns_b and not nouns_a & nouns_b:
        return False
    a, b = _normalize(a), _normalize(b)
    return _jaccard(a, b) >= 0.5 or len(a & b) / min(len(a), len(b)) >= 0.6


def _outfit_parts(tags: dict) -> tuple[set[str], set[str], set[str]]:
    main = _tokens(tags.get("onepiece", "")) or _tokens(tags.get("top", ""))
    return main, _tokens(tags.get("bottom", "")), _tokens(tags.get("outer", ""))


def same_outfit(a: dict, b: dict) -> bool:
    """Compare only what is visible in both photos; a dress never equals top + skirt. Photos carrying the same
    myshare outfit label (filename_outfit) are the same outfit whatever the wording of their descriptions."""
    label_a, label_b = a.get("filename_outfit"), b.get("filename_outfit")
    if label_a and label_a == label_b:
        return True
    main_a, bottom_a, outer_a = _outfit_parts(a)
    main_b, bottom_b, outer_b = _outfit_parts(b)
    if not _similar_garment(main_a, main_b):
        return False
    if bottom_a and bottom_b and not _similar_garment(bottom_a, bottom_b):
        return False
    if outer_a and outer_b and not _similar_garment(outer_a, outer_b):
        return False
    return True


def cluster_outfits(entries: list[tuple[str, dict]]) -> list[list[tuple[str, dict]]]:
    """Greedy clustering; richer descriptions (full body first) become cluster representatives."""
    def richness(entry):
        tags = entry[1]
        return (tags.get("visible") == "fullbody", sum(len(p) for p in _outfit_parts(tags)))

    clusters: list[list[tuple[str, dict]]] = []
    for entry in sorted(entries, key=richness, reverse=True):
        for cluster in clusters:
            if same_outfit(cluster[0][1], entry[1]):
                cluster.append(entry)
                break
        else:
            clusters.append([entry])
    return clusters


def outfit_description(tags: dict) -> str:
    parts = [tags.get(key, "") for key in ("onepiece", "top", "bottom", "outer")]
    return ", ".join(p for p in parts if p.strip().lower() not in _EMPTY and not p.lower().startswith("not visible"))


_GARMENT_COLORS = [
    ("dark brown", "진갈색"), ("light blue", "연파랑"), ("light pink", "연분홍"),
    ("off-white", "아이보리"), ("ivory", "아이보리"), ("cream", "크림색"),
    ("navy", "네이비"), ("burgundy", "버건디"), ("mustard", "머스타드"),
    ("black", "검정"), ("white", "흰색"), ("gray", "회색"), ("grey", "회색"),
    ("brown", "갈색"), ("red", "빨강"), ("blue", "파랑"), ("green", "초록"),
    ("pink", "분홍"), ("yellow", "노랑"), ("purple", "보라"), ("orange", "주황"),
    ("beige", "베이지"), ("gold", "금색"), ("silver", "은색"),
]
_GARMENT_PATTERNS = [
    ("cherry blossom", "벚꽃무늬"), ("floral", "꽃무늬"), ("flower", "꽃무늬"),
    ("striped", "스트라이프"), ("stripe", "스트라이프"), ("plaid", "체크"),
    ("checkered", "체크"), ("polka-dot", "도트"), ("polka dot", "도트"),
    ("patterned", "패턴"), ("print", "프린트"),
]
_GARMENT_DETAILS = [
    ("off-the-shoulder", "오프숄더"), ("off shoulder", "오프숄더"),
    ("long-sleeve", "긴팔"), ("long sleeve", "긴팔"), ("short-sleeve", "반팔"),
    ("short sleeve", "반팔"), ("sleeveless", "민소매"), ("strapless", "스트랩리스"),
    ("halter", "홀터넥"), ("cropped", "크롭"), ("crop", "크롭"), ("mini", "미니"),
    ("long", "롱"), ("flared", "플레어"), ("ruffle", "프릴"), ("high slit", "하이슬릿"),
    ("cutout", "컷아웃"), ("sheer", "시스루"), ("traditional", "전통"),
    ("lace", "레이스"), ("leather", "가죽"), ("denim", "데님"), ("satin", "새틴"),
    ("knit", "니트"), ("ribbed", "골지"),
]
_GARMENT_TYPES = [
    ("kimono dress", "기모노 원피스"), ("yukata dress", "유카타 원피스"),
    ("mini dress", "미니 원피스"), ("sundress", "선드레스"), ("one-piece", "원피스"),
    ("bodysuit", "보디수트"), ("jumpsuit", "점프수트"), ("romper", "롬퍼"),
    ("tank top", "탱크톱"), ("t-shirt", "티셔츠"), ("tee shirt", "티셔츠"),
    ("kimono", "기모노"), ("yukata", "유카타"), ("dress", "원피스"),
    ("cardigan", "가디건"), ("jacket", "재킷"), ("blazer", "블레이저"), ("coat", "코트"),
    ("sweater", "스웨터"), ("hoodie", "후드"), ("blouse", "블라우스"), ("shirt", "셔츠"),
    ("camisole", "캐미솔"), ("bra", "브라"), ("top", "상의"),
    ("mini skirt", "미니 스커트"), ("skirt", "스커트"), ("denim shorts", "데님 반바지"),
    ("shorts", "반바지"), ("jeans", "청바지"), ("trousers", "바지"), ("pants", "바지"),
    ("thong", "티팬티"), ("briefs", "브리프"), ("underwear", "속옷"), ("lingerie", "란제리"),
]


def _has_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _non_overlapping_labels(text: str, terms: list[tuple[str, str]]) -> list[str]:
    """Prefer longer phrases, then return their labels in source-text order without duplicates."""
    candidates = []
    for term, label in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        candidates.extend((match.start(), match.end(), label) for match in re.finditer(pattern, text))
    chosen: list[tuple[int, int, str]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if not any(candidate[0] < end and start < candidate[1] for start, end, _ in chosen):
            chosen.append(candidate)
    return list(dict.fromkeys(label for _, _, label in sorted(chosen)))


def garment_label(text: str) -> str:
    """Turn a model's compact English garment description into a concise Korean display label."""
    value = (text or "").strip().lower()
    if value in _EMPTY or value.startswith("not visible"):
        return ""
    colors = _non_overlapping_labels(value, _GARMENT_COLORS)
    pattern = next((label for term, label in _GARMENT_PATTERNS if _has_term(value, term)), "")
    details = _non_overlapping_labels(value, _GARMENT_DETAILS)
    garment = next((label for term, label in _GARMENT_TYPES if _has_term(value, term)), "의상")
    details = [label for label in details if label not in garment]
    parts = list(dict.fromkeys([*colors, pattern, *details, garment]))
    return " ".join(part for part in parts if part)


def outfit_label(tags: dict) -> str:
    parts = [garment_label(tags.get(key, "")) for key in ("onepiece", "top", "bottom", "outer")]
    return " · ".join(part for part in parts if part) or "기타 의상"


HAIR_LENGTH = [("short", "짧은"), ("bob", "짧은"), ("pixie", "짧은"), ("chin", "짧은"),
               ("medium", "중간"), ("shoulder", "중간"), ("long", "긴")]
HAIR_COLOR = [("dark brown", "진갈색"), ("black", "검정"), ("brown", "갈색"), ("chestnut", "적갈색"),
              ("auburn", "적갈색"), ("red", "적갈색"), ("blonde", "금발"), ("golden", "금발"),
              ("silver", "은발"), ("gray", "회색"), ("grey", "회색"), ("white", "흰색"), ("pink", "분홍"),
              ("blue", "파랑"), ("purple", "보라"), ("green", "초록")]
HAIR_TIED = ("ponytail", "braid", "bun", "tied", "up-do", "updo", "twin", "pigtail")
POSES = [("lying", "누워 있음"), ("reclin", "누워 있음"), ("kneel", "무릎 꿇음"), ("crouch", "쪼그려 앉음"),
         ("squat", "쪼그려 앉음"), ("sitting", "앉아 있음"), ("seated", "앉아 있음"), ("sit", "앉아 있음"),
         ("lean", "기대어 있음"), ("walk", "걷는 중"), ("stand", "서 있음")]


def hair_key(tags: dict) -> tuple[str, str, str]:
    text = (tags.get("hair") or "").lower()
    length = next((label for word, label in HAIR_LENGTH if word in text), "")
    color = next((label for word, label in HAIR_COLOR if word in text), "")
    style = "묶음" if any(word in text for word in HAIR_TIED) else ""
    return length, color, style


def hair_label(keys: tuple[str, str, str], members: list[dict]) -> str:
    length, color, style = keys
    texts = [(m.get("hair") or "").lower() for m in members]

    def majority(*words):
        return sum(any(w in t for w in words) for t in texts) * 2 >= len(texts)

    extras = [label for words, label in (
        (("wavy", "curl"), "웨이브"), (("straight",), "스트레이트"), (("bangs", "fringe"), "앞머리")) if majority(*words)]
    parts = [p for p in (f"{length} 머리" if length else "", color, style, *extras) if p]
    return " · ".join(parts) or "기타 헤어"


def pose_key(tags: dict) -> str:
    text = (tags.get("pose") or "").lower()
    return next((label for word, label in POSES if word in text), "기타 자세")


def build_albums(kind: str, entries: list[tuple[str, dict]]) -> list[dict]:
    """entries: [(file name, description)] -> [{label, count, names}] largest first."""
    groups: list[tuple[str, list[tuple[str, dict]]]] = []
    if kind == "outfit":
        for cluster in cluster_outfits(entries):
            groups.append((outfit_label(cluster[0][1]), cluster))
    elif kind == "hair":
        by_key: dict[tuple, list] = {}
        for entry in entries:
            by_key.setdefault(hair_key(entry[1]), []).append(entry)
        groups = [(hair_label(key, [t for _, t in members]), members) for key, members in by_key.items()]
    elif kind == "pose":
        by_pose: dict[str, list] = {}
        for entry in entries:
            by_pose.setdefault(pose_key(entry[1]), []).append(entry)
        groups = list(by_pose.items())
    else:
        raise ValueError("앨범 종류는 outfit, hair, pose 중 하나입니다.")
    result = [{"label": label, "count": len(members), "names": [name for name, _ in members]}
              for label, members in groups]
    return sorted(result, key=lambda g: -g["count"])


# --- duplicates and look-alikes ---

def dhash(path: Path) -> int:
    """64-bit difference hash of an image (gradient between neighbouring pixels of a 9x8 grayscale)."""
    with Image.open(path) as image:
        small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming(a: np.ndarray, value: int) -> np.ndarray:
    xor = (a ^ np.uint64(value)).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(xor, axis=1).sum(axis=1)


def find_duplicates(root: Path, paths: list[Path], thumb_fn, full_hash_fn, max_distance: int = 3) -> list[dict]:
    """Groups of identical files ("exact") and visually similar ones ("similar"), biggest first.

    Exact: same size and same full hash. Similar: dHash of the cached thumbnail within `max_distance`
    bits of any group member (single linkage). Exact groups are not repeated as similar groups."""
    names = {path: path.name for path in paths}
    by_size: dict[int, list[Path]] = {}
    for path in paths:
        by_size.setdefault(path.stat().st_size, []).append(path)
    parent = {path: path for path in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    exact_keys: dict[str, list[Path]] = {}
    content_hashes: dict[Path, str] = {}
    for same_size in by_size.values():
        if len(same_size) > 1:
            for path in same_size:
                digest = full_hash_fn(path)
                content_hashes[path] = digest
                exact_keys.setdefault(digest, []).append(path)
    exact_groups = [group for group in exact_keys.values() if len(group) > 1]
    for group in exact_groups:
        for path in group[1:]:
            parent[find(path)] = find(group[0])

    cache_file = state_dir(root) / "phash.json"
    with LOCK:
        cache = _read_json(cache_file, {})
    hashes = {}
    for path in paths:
        key = quick_hash(path)
        if key not in cache:
            cache[key] = format(dhash(thumb_fn(path)), "016x")
        hashes[path] = int(cache[key], 16)
    with LOCK:
        _write_json(cache_file, cache)

    values = np.array([hashes[p] for p in paths], dtype=np.uint64)
    for i, path in enumerate(paths[:-1]):
        close = np.nonzero(_hamming(values[i + 1:], hashes[path]) <= max_distance)[0]
        for j in close:
            parent[find(paths[i + 1 + j])] = find(path)

    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        grouped.setdefault(find(path), []).append(path)
    result = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        first_hash = content_hashes.get(members[0])
        is_exact = first_hash is not None and all(content_hashes.get(path) == first_hash for path in members)
        result.append({"kind": "exact" if is_exact else "similar", "count": len(members),
                       "names": [names[p] for p in members]})
    return sorted(result, key=lambda g: (g["kind"] != "exact", -g["count"]))


# --- myshare-style file names: <folder>_<outfit>_<shot>_<W>x<H>_<n>k_<seq>.<ext> ---

COMMON_RATIOS = ((1, 2), (9, 16), (4, 7), (3, 5), (5, 8), (2, 3), (5, 7), (3, 4), (4, 5), (5, 6), (7, 8), (1, 1),
                 (8, 7), (6, 5), (5, 4), (4, 3), (7, 5), (3, 2), (8, 5), (5, 3), (16, 9), (2, 1))
FINAL_NAME_RE = re.compile(r"^.+_(?P<outfit>[a-z0-9]+)_(?:headshot|upperbody|fullbody)_[0-9]+x[0-9]+_[1-4]k_[0-9]+\.[^.]+$",
                           re.IGNORECASE)
EDIT_SUFFIX_RE = re.compile(r"_edit[0-9]*(?=\.[^.]+$)", re.IGNORECASE)


def needs_name(name: str) -> bool:
    """True for files not yet in myshare form. Edited copies of a named file (<name>_edit.png) count as named,
    so renaming never splits them from their original."""
    return not FINAL_NAME_RE.fullmatch(EDIT_SUFFIX_RE.sub("", name))


def aspect_label(width: int, height: int, tolerance: float = 0.025) -> str:
    actual = width / height
    nearest = min(COMMON_RATIOS, key=lambda r: abs(math.log(actual / (r[0] / r[1]))))
    if abs(actual - nearest[0] / nearest[1]) / actual <= tolerance:
        return f"{nearest[0]}x{nearest[1]}"
    approx = Fraction(width, height).limit_denominator(20)
    return f"{approx.numerator}x{approx.denominator}"


def resolution_label(width: int, height: int) -> str:
    longest = max(width, height)
    return "4k" if longest >= 3840 else "3k" if longest >= 2880 else "2k" if longest >= 1920 else "1k"


def outfit_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())[:60]


def distinct_token(token: str, description: str, taken: set[str]) -> str:
    """A token not in `taken`, made of `token` plus words of the outfit description it does not name yet
    (colours first, then the rest in reading order); letters as the last resort."""
    words = [w for w in re.findall(r"[a-z0-9]+", (description or "").lower()) if w not in _STOP and w not in token]
    words.sort(key=lambda w: w not in COLOR_FAMILIES)  # stable: colours first, reading order kept
    candidate = token
    for word in words:
        candidate = outfit_token(word + candidate)[:40]
        if candidate not in taken:
            return candidate
    for letter in "bcdefghijklmnopqrstuvwxyz":
        if (candidate + letter)[:40] not in taken:
            return (candidate + letter)[:40]
    return candidate


def prefix_token(folder_name: str) -> str:
    return "".join(c.lower() for c in folder_name.strip() if c.isalnum() or c in "_-").strip("_-") or "img"


def known_outfits(directory: Path) -> list[str]:
    return sorted({m.group("outfit").lower() for p in directory.iterdir()
                   if p.is_file() and (m := FINAL_NAME_RE.fullmatch(p.name))})


def plan_names(directory: Path, items: list[dict]) -> list[dict]:
    """items: [{name, outfit, shot}] -> [{name, new_name}], continuing the sequence numbers already in the folder.
    Items whose outfit token is empty or shot is unknown are skipped."""
    prefix = prefix_token(directory.name)
    taken = {p.name for p in directory.iterdir()}
    next_seq: dict[str, int] = {}
    plan = []
    for item in items:
        outfit, shot = outfit_token(item["outfit"]), item["shot"]
        if len(outfit) < 2 or shot not in ("headshot", "upperbody", "fullbody"):
            continue
        source = directory / item["name"]
        with Image.open(source) as image:
            width, height = image.size
        group = f"{outfit}_{shot}_{aspect_label(width, height)}_{resolution_label(width, height)}"
        pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(group)}_([0-9]+)\.[^.]+$")
        if group not in next_seq:
            next_seq[group] = max((int(m.group(1)) for n in taken if (m := pattern.fullmatch(n))), default=0) + 1
        new_name = f"{prefix}_{group}_{next_seq[group]:02d}{source.suffix.lower()}"
        next_seq[group] += 1
        taken.add(new_name)
        plan.append({"name": item["name"], "new_name": new_name})
    return plan


def apply_rename(root: Path, directory: Path, plan: list[dict]) -> dict:
    """Two-phase rename (temporary names first) so a failure rolls everything back."""
    with FILE_OP_LOCK:
        staged, done = [], []
        try:
            for index, item in enumerate(plan):
                source = directory / item["name"]
                _check_movable(root, source)
                if (directory / item["new_name"]).exists():
                    raise ValueError(f"이미 같은 이름이 있습니다: {item['new_name']}")
                temporary = directory / f".__rename_{uuid.uuid4().hex[:8]}_{index}{source.suffix.lower()}"
                source.rename(temporary)
                staged.append((temporary, source, directory / item["new_name"]))
            for temporary, source, target in staged:
                temporary.rename(target)
                done.append((target, source))
            return _record(root, "rename", [{"from": _rel(root, s), "to": _rel(root, t)} for t, s in done])
        except Exception:
            for target, source in reversed(done):
                if target.exists():
                    target.rename(source)
            for temporary, source, _ in staged:
                if temporary.exists():
                    temporary.rename(source)
            raise
