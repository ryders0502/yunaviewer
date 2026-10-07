#!/usr/bin/env python3
"""Image viewer with AI search. Settings in yunaviewer.cfg; python3 server.py [ROOT] [--port N]"""

from __future__ import annotations

import argparse
import functools
import hashlib
from concurrent.futures import ThreadPoolExecutor
import configparser
import datetime
import http.server
import io
import json
import mimetypes
import os
import re
from PIL import ExifTags
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps, ImageStat

import effects
import library
from library import IMAGE_EXTS

THUMB_SIZE = 400
REF_SIZE = 768
MAX_CANDIDATES = 3000  # ponytail: all tags of a folder go into one text prompt, chunk it if folders grow past this
DESCRIBE_CHUNK_SIZE = 8  # images per describe call
VERIFY_CHUNK_SIZE = 24  # images per verify call
TEXT_CHUNK_SIZE = 100  # descriptions per text-search call; flash-lite misses matches in longer lists
VERIFY_LAYOUT = (4, 2, 480, 600)  # per sheet, columns, cell width, cell height
INDEX_HTML = Path(__file__).with_name("index.html")
AGENT_SCRIPT = Path(__file__).with_name("search_agent.py")
CONFIG_PATH = Path(os.getenv("YUNAVIEWER_AGENT_CONFIG", str(Path(__file__).with_name("yunaviewer.cfg"))))
CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_PATH, encoding="utf-8")
PARALLEL_CALLS = CONFIG.getint("llm", "parallel_calls", fallback=2)
DESCRIBE_MODEL = CONFIG.get("llm", "describe_model", fallback="")
SEARCH_MODEL = CONFIG.get("llm", "search_model", fallback="")
ROOT = Path.home() / "Pictures"
TAGS_LOCK = threading.Lock()  # guards the tags file and TAG_STATUS
TAGGING_LOCK = threading.Lock()  # ponytail: one folder is described at a time, per-folder locks if several are opened at once
TAG_STATUS: dict[str, dict] = {}
_size_cache: dict[tuple[str, int], tuple[int, int]] = {}
THUMB_LOCKS = [threading.Lock() for _ in range(64)]  # striped by file: generating one thumbnail never blocks cached ones


def safe_path(rel: str) -> Path:
    """Resolve a virtual path without losing a top-level directory-symlink alias.

    A directory symlink deliberately placed directly under ROOT (for example Pictures/myshare)
    is treated as a mounted library folder. Traversal outside ROOT or outside that symlink's own
    target remains forbidden. Returning the lexical path keeps operation logs and URLs relative
    to ROOT while filesystem calls still follow the symlink normally.
    """
    relative = Path(rel.lstrip("/"))
    if ".." in relative.parts:
        raise PermissionError(rel)
    path = ROOT / relative
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        return path
    if relative.parts:
        mount = ROOT / relative.parts[0]
        if mount.is_symlink():
            target = mount.resolve()
            if target.is_dir() and resolved.is_relative_to(target):
                return path
    raise PermissionError(rel)


def image_info(path: Path) -> tuple[int, int, float | None]:
    """(width, height, EXIF capture time as a timestamp or None), cached per file version."""
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _size_cache:
        taken = None
        with Image.open(path) as img:
            w, h = img.size
            # Skip PNG: getexif() decodes the whole image there (and AI PNGs carry no capture time anyway)
            if img.format != "PNG":
                exif = img.getexif()
                if exif.get(0x0112, 1) in (5, 6, 7, 8):  # EXIF orientation 5-8 swaps width/height
                    w, h = h, w
                stamp = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)  # DateTimeOriginal, else DateTime
                try:
                    taken = datetime.datetime.strptime(str(stamp).strip("\x00 "), "%Y:%m:%d %H:%M:%S").timestamp()
                except (TypeError, ValueError):
                    pass
        _size_cache[key] = (w, h, taken)
    return _size_cache[key]


def image_size(path: Path) -> tuple[int, int]:
    return image_info(path)[:2]


def image_metadata(rel: str) -> dict:
    path = safe_path(rel)
    if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
        raise FileNotFoundError(rel)
    stat = path.stat()
    data = {"name": path.name, "width": None, "height": None, "bytes": stat.st_size,
            "modified": stat.st_mtime, "camera": "", "taken": "", "make": "", "model": ""}
    with Image.open(path) as image:
        data["width"], data["height"] = image_size(path)
        try:
            exif = {ExifTags.TAGS.get(key, str(key)): value for key, value in image.getexif().items()}
            data["make"] = str(exif.get("Make", "")).strip()
            data["model"] = str(exif.get("Model", "")).strip()
            data["camera"] = " ".join(part for part in (data["make"], data["model"]) if part)
            data["taken"] = str(exif.get("DateTimeOriginal") or exif.get("DateTime") or "")
        except (OSError, TypeError, ValueError):
            pass
        data["generation"] = library.generation_info(image)
    data["analysis"] = stored_tags([path]).get(path, {})
    return data


def list_dir(rel: str) -> dict:
    directory = safe_path(rel)
    if directory == ROOT / library.STATE_DIR / "trash":  # created on the first delete; until then show it empty
        directory.mkdir(parents=True, exist_ok=True)
    favorites = library.load_favorites(ROOT)
    marks = library.load_marks(ROOT)
    dirs, files = [], []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            dirs.append(entry.name)
        elif entry.suffix.lower() in IMAGE_EXTS:
            try:
                w, h, taken = image_info(entry)
                stat = entry.stat()
                key = library.quick_hash(entry)
            except OSError:
                continue
            # taken: EXIF capture time, or the file time when there is none (sorting needs a value)
            item = {"name": entry.name, "w": w, "h": h, "mtime": stat.st_mtime, "bytes": stat.st_size,
                    "taken": taken or stat.st_mtime, "fav": key in favorites}
            item.update(marks.get(key, {}))  # rating, note (only when set)
            files.append(item)
    files.sort(key=lambda f: f["mtime"], reverse=True)  # newest first
    rel_dir = directory.relative_to(ROOT).as_posix()
    trash = ROOT / library.STATE_DIR / "trash"
    trashed = sum(1 for p in trash.iterdir() if p.is_file()) if trash.is_dir() else 0  # for the 🗑 button badge
    return {"dir": "" if rel_dir == "." else rel_dir, "dirs": dirs, "files": files, "trashed": trashed}


def thumbnail(path: Path) -> Path:
    relative = path.relative_to(ROOT).as_posix()
    cache_key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    cache = ROOT / ".yunaviewer" / "thumbs" / f"{cache_key}.jpg"
    source_mtime = path.stat().st_mtime_ns

    def fresh() -> bool:
        try:
            return cache.stat().st_mtime_ns == source_mtime
        except FileNotFoundError:
            return False

    if fresh():
        return cache
    with THUMB_LOCKS[int(cache_key[:8], 16) % len(THUMB_LOCKS)]:
        if not fresh():  # another request may have produced it while we waited
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}.tmp")
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    img.thumbnail((THUMB_SIZE, THUMB_SIZE * 3))
                    img.save(temporary, "JPEG", quality=85)
                os.utime(temporary, ns=(source_mtime, source_mtime))
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
    return cache


def make_contact_sheets(paths: list[Path], ids: list[str], job_dir: Path,
                        layout: tuple[int, int, int, int] = VERIFY_LAYOUT) -> list[str]:
    sheets = []
    per_sheet, columns, cell_w, cell_h = layout
    font = ImageFont.load_default(size=18)
    for page_start in range(0, len(paths), per_sheet):
        page = list(zip(paths, ids))[page_start:page_start + per_sheet]
        rows = (len(page) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "#181818")
        draw = ImageDraw.Draw(canvas)
        for slot, (path, image_id) in enumerate(page):
            with Image.open(path) as opened:
                thumb = ImageOps.contain(ImageOps.exif_transpose(opened).convert("RGB"),
                                         (cell_w - 12, cell_h - 42))
            x = (slot % columns) * cell_w
            y = (slot // columns) * cell_h
            canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 34))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#555")
            draw.text((x + 8, y + 7), image_id, fill="white", font=font)
        name = f"contact_{len(sheets) + 1:02d}.jpg"
        canvas.save(job_dir / name, "JPEG", quality=88, optimize=True)
        sheets.append(name)
    return sheets


def list_candidates(rel_dir: str, selected: list[str], within: list[str] | None = None) -> tuple[list[Path], list[Path]]:
    directory = safe_path(rel_dir)
    selected_paths = [safe_path(rel) for rel in selected]
    if any(not p.is_file() or p.suffix.lower() not in IMAGE_EXTS for p in selected_paths):
        raise ValueError("선택한 항목 중 이미지가 아닌 것이 있습니다.")
    candidates = sorted(
        (p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p not in selected_paths
         and (within is None or p.name in within)),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        raise ValueError("검색할 이미지가 없습니다.")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"한 번에 최대 {MAX_CANDIDATES}장까지 검색할 수 있습니다.")
    return candidates, selected_paths


def save_refs(selected_paths: list[Path], job_dir: Path) -> list[str]:
    refs = []
    for index, path in enumerate(selected_paths, 1):
        with Image.open(path) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
            img.thumbnail((REF_SIZE, REF_SIZE))
            img.save(job_dir / f"ref_{index:02d}.jpg", "JPEG", quality=88)
        refs.append(f"ref_{index:02d}.jpg")
    return refs


# myshare names organized photos <prefix>_<outfit>_<shot>_<W>x<H>_<n>k_<seq>.<ext>
OUTFIT_NAME_RE = re.compile(r"_(?P<outfit>[a-z0-9]+)_(?:headshot|upperbody|fullbody)_[0-9]+x[0-9]+_[1-4]k_[0-9]+\.[^.]+$",
                            re.IGNORECASE)


def search_tags(path: Path, tags: dict) -> dict:
    """Stored description plus the outfit token from a myshare filename, if any."""
    match = OUTFIT_NAME_RE.search(path.name)
    return {**tags, "filename_outfit": match.group("outfit").lower()} if match else tags


def normalize_id(value) -> str:
    """Models sometimes answer '99' or 'img-099' instead of 'IMG-099'."""
    digits = re.search(r"\d+", str(value))
    return f"IMG-{int(digits.group()):03d}" if digits else ""


def map_matches(response: dict, id_map: dict, selected: list[str]) -> list[str]:
    names = [Path(rel).name for rel in selected]
    for image_id in response.get("matches", []):
        name = id_map.get(normalize_id(image_id))
        if name and name not in names:
            names.append(name)
    return names


def invoke_agent(job_dir: Path, request: dict) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    request_path = job_dir / "agent_request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    python = Path(__file__).parent / os.getenv("YUNAVIEWER_AGENT_PYTHON", CONFIG.get(
        "agent", "python", fallback=".venv/bin/python"
    ))
    completed = subprocess.run(
        [python, str(AGENT_SCRIPT), str(request_path)], capture_output=True, text=True,
        timeout=CONFIG.getint("agent", "timeout", fallback=600), check=False,
        env={**os.environ, "YUNAVIEWER_AGENT_CONFIG": str(CONFIG_PATH)},
    )
    if completed.returncode != 0:
        if "RESOURCE_EXHAUSTED" in completed.stderr:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        message = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error"
        raise RuntimeError(f"검색 에이전트 실패: {message}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("검색 에이전트가 결과를 반환하지 않았습니다.")
    return json.loads(lines[-1])


def verified_names(response: dict, id_map: dict) -> set[str]:
    return {id_map[key] for verdict in response.get("verdicts", [])
            if verdict.get("match") is True and (key := normalize_id(verdict.get("id"))) in id_map}


def call_agent(job_dir: Path, request: dict) -> dict:
    for attempt in range(4):
        try:
            return invoke_agent(job_dir, request)
        except RuntimeError as error:
            if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
                raise RuntimeError("Gemini 무료 사용 한도를 초과했습니다. 잠시 후(일일 한도면 내일) 다시 시도하세요.")
            if attempt == 3:
                raise
            # Repeating the same large describe sheet rarely repairs a forced-tool failure. Let the
            # caller split that batch after one retry instead of spending all four calls on it.
            if "StructuredOutputException" in str(error) and attempt >= 1:
                raise
            # 503 = model overloaded (common on free tier), back off; other errors (bad structured output) retry now
            if "503" in str(error) or "UNAVAILABLE" in str(error):
                time.sleep(15 * (attempt + 1))


def chunk_request(paths: list[Path], job_dir: Path, extra: dict) -> tuple[dict, dict]:
    ids = [f"IMG-{i:03d}" for i in range(1, len(paths) + 1)]
    job_dir.mkdir(parents=True, exist_ok=True)
    request = {"image_ids": ids, "references": [], "contact_sheets": make_contact_sheets(paths, ids, job_dir), **extra}
    return request, dict(zip(ids, paths))


# --- per-image descriptions (tags), stored in ROOT/.yunaviewer/tags.json keyed by file content hash,
# so moved or renamed images keep their tags ---

_hash_cache: dict[tuple[str, int, int], str] = {}


def file_hash(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hash_cache:
        with path.open("rb") as handle:
            _hash_cache[key] = hashlib.file_digest(handle, "sha256").hexdigest()
    return _hash_cache[key]


def tags_path() -> Path:
    return ROOT / ".yunaviewer" / "tags.json"


def index_status_path() -> Path:
    return ROOT / ".yunaviewer" / "index_status.json"


def write_tags(tags: dict) -> None:
    tags_path().parent.mkdir(parents=True, exist_ok=True)
    temporary = tags_path().with_suffix(".tmp")
    temporary.write_text(json.dumps(tags, ensure_ascii=False), encoding="utf-8")
    temporary.replace(tags_path())


def migrate_path_keys(old: dict) -> dict:
    """Old format was {relative path: {mtime_ns, tags}}. mv keeps mtime, so files moved since then are
    found again by their mtime."""
    by_mtime = {entry["mtime_ns"]: entry["tags"] for entry in old.values()}
    tags = {}
    for path in ROOT.rglob("*"):
        if (".yunaviewer" not in path.parts and path.suffix.lower() in IMAGE_EXTS and path.is_file()
                and (value := by_mtime.get(path.stat().st_mtime_ns)) is not None):
            tags[file_hash(path)] = value
    write_tags(tags)
    return tags


def load_tags() -> dict:
    """Caller holds TAGS_LOCK."""
    try:
        tags = json.loads(tags_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if any("mtime_ns" in value for value in tags.values()):
        tags = migrate_path_keys(tags)
    return tags


def stored_tags(paths: list[Path]) -> dict[Path, dict]:
    with TAGS_LOCK:
        tags = load_tags()
    return {path: tags[digest] for path in paths if (digest := file_hash(path)) in tags}


def store_tags(new: dict[Path, dict]) -> None:
    with TAGS_LOCK:
        tags = load_tags()
        tags.update({file_hash(path): value for path, value in new.items()})
        write_tags(tags)


def folder_images(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def describe_images(paths: list[Path], status: dict) -> None:
    """Describe images without valid tags, saving after every chunk so progress survives restarts."""
    existing = stored_tags(paths)
    missing = [p for p in paths if p not in existing]
    status.update(total=len(paths), done=len(paths) - len(missing), error="")
    job_root = ROOT / ".yunaviewer" / "jobs" / f"describe_{uuid.uuid4().hex}"
    stop = threading.Event()

    def describe_batch(batch: list[Path], batch_name: str) -> None:
        if stop.is_set():
            return
        try:
            job_dir = job_root / batch_name
            request, id_map = chunk_request(batch, job_dir, {"mode": "describe", "model": DESCRIBE_MODEL})
            response = call_agent(job_dir, request)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            # Gemini occasionally refuses the structured-output tool for a crowded contact sheet.
            # Smaller sheets usually recover it, while successful halves are saved independently.
            if "StructuredOutputException" in str(error) and len(batch) > 1 and not stop.is_set():
                middle = len(batch) // 2
                describe_batch(batch[:middle], f"{batch_name}_a")
                describe_batch(batch[middle:], f"{batch_name}_b")
                return
            detail = f"{error} (이미지: {batch[0].name})" if len(batch) == 1 else str(error)
            with TAGS_LOCK:
                status["error"] = detail
            if "한도" in str(error):
                stop.set()  # quota exhausted: stop now, the next folder open resumes
            return
        new = {}
        for entry in response.get("images", []):
            path = id_map.get(normalize_id(entry.pop("id", "")))
            if path:
                new[path] = entry
        store_tags(new)
        with TAGS_LOCK:
            status["done"] += len(new)

    def run(item: tuple[int, list[Path]]) -> None:
        describe_batch(item[1], f"{item[0]:03d}")

    chunks = [missing[i:i + DESCRIBE_CHUNK_SIZE] for i in range(0, len(missing), DESCRIBE_CHUNK_SIZE)]
    try:
        with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
            list(pool.map(run, enumerate(chunks)))
    finally:
        shutil.rmtree(job_root, ignore_errors=True)


def tag_status(directory: Path) -> dict:
    key = directory.relative_to(ROOT).as_posix()
    with TAGS_LOCK:
        if key not in TAG_STATUS:
            try:
                saved = json.loads(index_status_path().read_text(encoding="utf-8")).get(key, {})
            except (FileNotFoundError, json.JSONDecodeError):
                saved = {}
            TAG_STATUS[key] = {"running": False, "total": 0, "done": 0, "error": "",
                               "last_success": saved.get("last_success")}
        status = TAG_STATUS[key]
    return status


def tag_status_snapshot(directory: Path) -> dict:
    """Return current progress, recalculating idle folders so completion survives a server restart."""
    status = tag_status(directory)
    with TAGS_LOCK:
        snapshot = dict(status)
    if snapshot["running"]:
        return snapshot
    paths = folder_images(directory)
    done = len(stored_tags(paths))
    with TAGS_LOCK:
        if not status["running"]:  # indexing may have started while hashes were being checked
            status.update(total=len(paths), done=done)
            if done == len(paths):
                status["error"] = ""
        return dict(status)


def start_tagging(directory: Path) -> None:
    """Describe new or changed images after an explicit indexing or search action."""
    status = tag_status(directory)
    with TAGS_LOCK:
        if status["running"]:
            return
        status["running"] = True

    def work() -> None:
        try:
            with TAGGING_LOCK:
                describe_images(folder_images(directory), status)
        finally:
            with TAGS_LOCK:
                status["running"] = False
                if status.get("done") == status.get("total") and not status.get("error"):
                    status["last_success"] = time.time()
                    try:
                        saved = json.loads(index_status_path().read_text(encoding="utf-8"))
                    except (FileNotFoundError, json.JSONDecodeError):
                        saved = {}
                    saved[directory.relative_to(ROOT).as_posix()] = {"last_success": status["last_success"]}
                    index_status_path().parent.mkdir(parents=True, exist_ok=True)
                    temporary = index_status_path().with_suffix(".tmp")
                    temporary.write_text(json.dumps(saved), encoding="utf-8")
                    temporary.replace(index_status_path())

    threading.Thread(target=work, daemon=True).start()


class NeedsIndex(RuntimeError):
    """Some photos have no stored description yet and the caller has not agreed to send them to the model."""

    def __init__(self, count: int):
        super().__init__(f"AI 분석이 아직 안 된 이미지가 {count}장 있습니다.")
        self.count = count


def ensure_described(paths: list[Path], directory: Path, allow_index: bool) -> None:
    """Photos are sent to the model only on an explicit request: without allow_index, missing descriptions
    raise NeedsIndex (nothing is uploaded) so the UI can ask the user first."""
    existing = stored_tags(paths)
    missing = [p for p in paths if p not in existing]
    if not missing:
        return
    if not allow_index:
        raise NeedsIndex(len(missing))
    with TAGGING_LOCK:
        describe_images(paths, tag_status(directory))


MAX_ASK_IMAGES = 4
SEARCH_WORDS = re.compile(r"찾|검색|보여|골라|고르|모아|추려|걸러|필터|find|search|show me", re.IGNORECASE)


def is_question(prompt: str, selected: list[str]) -> bool:
    """A prompt about the selected photos themselves ("두 사진 인물이 얼마나 비슷해?") rather than a search."""
    return bool(selected) and not SEARCH_WORDS.search(prompt)


def ask_images(rel_dir: str, prompt: str, selected: list[str]) -> dict:
    """Answer a free question about the selected photos. Only those photos are sent (small JPEGs), and only
    because the user selected them and asked; nothing is indexed."""
    paths = [safe_path(rel) for rel in selected]
    if any(not p.is_file() or p.suffix.lower() not in IMAGE_EXTS for p in paths):
        raise ValueError("선택한 항목 중 이미지가 아닌 것이 있습니다.")
    if not paths:
        raise ValueError("선택한 이미지를 찾지 못했습니다.")
    if len(paths) > MAX_ASK_IMAGES:
        raise ValueError(f"질문은 이미지 {MAX_ASK_IMAGES}장까지 선택해서 할 수 있습니다.")
    job_dir = ROOT / ".yunaviewer" / "jobs" / uuid.uuid4().hex
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        refs = save_refs(paths, job_dir)
        response = call_agent(job_dir, {"mode": "ask", "model": SEARCH_MODEL, "references": refs, "prompt": prompt})
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"answer": response.get("answer", ""), "images": [p.name for p in paths]}


def run_search(rel_dir: str, prompt: str, selected: list[str], allow_index: bool = False,
               within: list[str] | None = None, previous: str = "") -> dict:
    """within: search only these file names (the results on screen); previous: the request that produced them,
    so a follow-up like "그중 앉아 있는 것만" is understood as narrowing it."""
    candidates, selected_paths = list_candidates(rel_dir, selected, set(within) if within is not None else None)
    if previous:
        prompt = (f"{prompt}\n(Follow-up: the candidates are the results of the previous request "
                  f"\"{previous}\"; keep only those that also satisfy this new request.)")
    job_root = ROOT / ".yunaviewer" / "jobs" / uuid.uuid4().hex
    try:
        ensure_described(candidates + selected_paths, safe_path(rel_dir), allow_index)
        tags = stored_tags(candidates + selected_paths)
        if any(p not in tags for p in selected_paths):
            raise RuntimeError(tag_status(safe_path(rel_dir))["error"] or "선택한 이미지를 분석하지 못했습니다.")
        tagged = [p for p in candidates if p in tags]
        untagged = [p for p in candidates if p not in tags]
        if len(untagged) > VERIFY_CHUNK_SIZE * 2:
            raise RuntimeError(tag_status(safe_path(rel_dir))["error"] or "이미지 분석이 끝나지 않았습니다.")

        # stage 1: text search over stored descriptions, in parallel chunks
        ref_tags = {f"REF-{i}": search_tags(p, tags[p]) for i, p in enumerate(selected_paths, 1)}
        job_root.mkdir(parents=True, exist_ok=True)

        def text_search(chunk: list[Path]) -> tuple[dict, dict]:
            ids = [f"IMG-{i:03d}" for i in range(1, len(chunk) + 1)]
            return call_agent(job_root / uuid.uuid4().hex, {
                "prompt": prompt, "model": SEARCH_MODEL, "references": [], "reference_tags": ref_tags,
                "candidate_tags": {image_id: search_tags(path, tags[path]) for image_id, path in zip(ids, chunk)},
            }), {image_id: path.name for image_id, path in zip(ids, chunk)}

        chunks = [tagged[i:i + TEXT_CHUNK_SIZE] for i in range(0, len(tagged), TEXT_CHUNK_SIZE)]
        with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
            results = list(pool.map(text_search, chunks))
        found = []
        for chunk_response, chunk_map in results:
            found += map_matches(chunk_response, chunk_map, [])
        response = next((r for r, _ in results if r.get("matches")), results[0][0] if results else {})
        # images the describer failed on go straight to the image check
        # cheap recall boost; the image check below drops wrong ones (e.g. a mislabeled filename)
        if ref_tags:
            labels = {t["filename_outfit"] for t in ref_tags.values() if "filename_outfit" in t}
            garments = {(t.get("top"), t.get("onepiece")) for t in ref_tags.values()}
            for path in tagged:
                described = search_tags(path, tags[path])
                if (described.get("filename_outfit") in labels
                        or (described.get("top"), described.get("onepiece")) in garments):
                    found.append(path.name)
        found += [p.name for p in untagged]
        found = list(dict.fromkeys(found))
        # with references, the target is their stored description; the model's own summary can drift
        summary = (json.dumps(ref_tags, ensure_ascii=False) if ref_tags
                   else response.get("reference_summary", ""))

        # stage 2: per-image verdict on large images drops look-alikes
        by_name = {path.name: path for path in candidates}
        kept = set()
        if found and (selected_paths or summary):
            refs = save_refs(selected_paths, job_root)
            paths = [by_name[n] for n in found]

            def verify(item: tuple[int, list[Path]]) -> set[str]:
                job_dir = job_root / f"verify_{item[0]:02d}"
                request, chunk_map = chunk_request(item[1], job_dir, {
                    "mode": "verify", "prompt": prompt, "summary": summary, "model": SEARCH_MODEL})
                for ref in refs:
                    shutil.copy(job_root / ref, job_dir / ref)
                request["references"] = refs
                return verified_names(call_agent(job_dir, request), {k: v.name for k, v in chunk_map.items()})

            chunks = [paths[i:i + VERIFY_CHUNK_SIZE] for i in range(0, len(paths), VERIFY_CHUNK_SIZE)]
            with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
                for names in pool.map(verify, enumerate(chunks)):
                    kept |= names
        matches = [Path(rel).name for rel in selected] + [name for name in found if name in kept]
        return {"matches": matches, "message": response.get("message", "")}
    finally:
        shutil.rmtree(job_root, ignore_errors=True)


# --- image editing: one Pillow pipeline renders both the preview and the saved file ---

EDIT_RANGES = {"brightness": (0, 3), "contrast": (0, 3), "saturation": (0, 3), "sharpness": (0, 3),
               "gamma": (0.2, 5), "temperature": (-100, 100)}
PREVIEW_SIDE = 1400


# ponytail: in-memory cache of full-resolution body warps (a few seconds each), lost on restart and
# limited to 4 images; a disk cache keyed the same way if that becomes a bother
def load_base(path_str: str, rot: int, flip: bool) -> Image.Image:
    """The photo as the editor sees it: EXIF-upright, then the user's rotation/flip."""
    with Image.open(path_str) as opened:
        img = ImageOps.exif_transpose(opened)
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return effects.orient(img, rot, flip)


@functools.lru_cache(maxsize=4)
def body_warped(path_str: str, mtime_ns: int, rot: int, flip: bool, body_json: str) -> Image.Image:
    import numpy as np
    import body_edit  # mediapipe/opencv live in .venv only, so import on first use

    img = load_base(path_str, rot, flip)
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    rgb = np.array(img.convert("RGB"))
    try:
        warped, _, trim = body_edit.edit_image_array(rgb, json.loads(body_json))
    except body_edit.DetectionError as error:
        raise ValueError(f"체형 보정 실패: {error}") from None
    result = Image.fromarray(warped)
    if alpha is not None:  # ponytail: alpha is not warped with the body; fine for opaque photos
        result.putalpha(alpha)
    result.info["trim_bottom"] = trim  # rows to cut off at the bottom (see body_edit._ground_points)
    return result


def merge_body(current: list, update: list) -> list:
    """Overwrite the accumulated per-part totals with a command's new totals, clamped to the engine limit.
    (The model returns totals, not deltas: if it echoes an untouched part, nothing changes.)"""
    import body_edit

    total: dict[str, float] = {}
    for edit in list(current or []) + list(update or []):
        part = edit.get("part")
        if part not in body_edit.EDITABLE_PARTS:
            raise ValueError(f"지원하지 않는 체형 부위입니다: {part}")
        total[part] = float(edit.get("pct", 0))
    limit = body_edit.MAX_EDIT_PCT
    merged = [{"part": part, "pct": round(max(-limit, min(limit, pct)), 2)}
              for part, pct in total.items() if round(pct, 2)]
    return body_edit.validate_edits(merged) if merged else []


@functools.lru_cache(maxsize=2)
def detection(path_str: str, mtime_ns: int, rot: int, flip: bool, body_json: str):
    """(person mask, face landmarks or None) of the oriented, body-warped photo at full resolution."""
    import numpy as np
    import body_edit

    img = body_warped(path_str, mtime_ns, rot, flip, body_json) if body_json != "[]" else load_base(path_str, rot, flip)
    rgb = np.ascontiguousarray(np.array(img.convert("RGB")))
    try:
        with body_edit.LOCK:
            _, mask, face = body_edit.detect(rgb)
    except body_edit.DetectionError:
        raise ValueError("사람을 인식하지 못해 이 보정을 할 수 없습니다.") from None
    return mask[:rgb.shape[0], :rgb.shape[1]], face


@functools.lru_cache(maxsize=4)
def face_points(path_str: str, mtime_ns: int, rot: int, flip: bool, body_json: str):
    """Face landmarks of the oriented, body-warped photo (face model only, so close-ups work), or None."""
    import numpy as np
    import body_edit

    img = body_warped(path_str, mtime_ns, rot, flip, body_json) if body_json != "[]" else load_base(path_str, rot, flip)
    with body_edit.LOCK:
        return body_edit.detect_face(np.ascontiguousarray(np.array(img.convert("RGB"))))


def shape_params(params: dict) -> dict:
    shape = params.get("shape") or {}
    return {key: round(max(-effects.MAX_SHAPE_PCT, min(effects.MAX_SHAPE_PCT, float(shape[key]))), 2)
            for key in effects.SHAPE_KEYS if shape.get(key)}


FACE_RANGES = {"brightness": (0.5, 1.8, 1.0), "tone": (-50, 50, 0.0), "smooth": (0.0, 1.0, 0.0)}


def face_params(params: dict) -> dict:
    face = params.get("face") or {}
    return {key: min(max(float(face.get(key, default)), lo), hi) for key, (lo, hi, default) in FACE_RANGES.items()}


def background_params(params: dict) -> tuple[str, float]:
    bg = params.get("bg") or {}
    mode = bg.get("mode", "none")
    if mode not in effects.BG_MODES:
        mode = "none"
    return mode, min(max(float(bg.get("amount", 0.5)), 0.0), 1.0)


def orient_params(params: dict) -> tuple[int, bool]:
    orient = params.get("orient") or {}
    rot = int(orient.get("rot", 0)) % 360
    return (rot if rot in (0, 90, 180, 270) else 0), bool(orient.get("flip"))


def edit_image(path: Path, params: dict, max_side: int | None = None) -> Image.Image:
    rot, flip = orient_params(params)
    body = params.get("body") or []
    body_json = json.dumps(body, sort_keys=True)
    key = (str(path), path.stat().st_mtime_ns, rot, flip)
    img = body_warped(*key, body_json).copy() if body else load_base(str(path), rot, flip)
    trim = img.info.get("trim_bottom", 0) if body else 0
    shape = shape_params(params)
    if shape:  # before tone/background, whose masks come from the unshaped photo (a few px off at most)
        face_px = face_points(*key, body_json)
        reason = effects.face_too_small(face_px, img.height)
        if reason:
            raise ValueError(reason)
        img = effects.reshape_face(img, face_px, shape)
    erase = params.get("erase") or []
    if erase:  # painted spots, in the same pixels as the crop (oriented, body-warped photo)
        if not isinstance(erase, list) or not all(isinstance(s, dict) for s in erase):
            raise ValueError("지우기 영역 형식이 올바르지 않습니다.")
        img = effects.erase_spots(img, erase)
    face = face_params(params)
    bg_mode, bg_amount = background_params(params)
    face_changed = any(face[k] != default for k, (_, _, default) in FACE_RANGES.items())
    if face_changed or bg_mode != "none":
        mask, face_px = detection(*key, body_json)
        if face_changed:
            if face_px is None:
                raise ValueError("얼굴을 인식하지 못해 얼굴 보정을 할 수 없습니다.")
            img = effects.adjust_face(img, face_px, **face)
        if bg_mode != "none":
            img = effects.apply_background(img, mask, bg_mode, bg_amount)
    crop = params.get("crop") or {}
    x = min(max(int(crop.get("x", 0)), 0), img.width - 1)
    y = min(max(int(crop.get("y", 0)), 0), img.height - 1)
    w = min(max(int(crop.get("w", img.width)), 1), img.width - x)
    h = min(max(int(crop.get("h", img.height)), 1), img.height - y)
    size = params.get("size") or {}
    out = (min(max(int(size.get("w", w)), 1), 12000), min(max(int(size.get("h", h)), 1), 12000))
    if trim and y + h > img.height - trim:  # the crop reaches the empty band under lifted feet: cut it, keep the stretch
        cut = max(1, img.height - trim - y)
        out = (out[0], max(1, round(out[1] * cut / h)))
        h = cut
    img = img.crop((x, y, x + w, y + h))
    if max_side:  # preview: same aspect as the output, scaled down
        scale = min(1, max_side / max(out))
        out = (max(1, round(out[0] * scale)), max(1, round(out[1] * scale)))
    if out != img.size:
        img = img.resize(out, Image.Resampling.LANCZOS)

    value = {key: min(max(float(params.get(key, 0 if key == "temperature" else 1)), lo), hi)
             for key, (lo, hi) in EDIT_RANGES.items()}
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    rgb = img.convert("RGB")
    for key, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast),
                          ("saturation", ImageEnhance.Color)):
        if value[key] != 1:
            rgb = enhancer(rgb).enhance(value[key])
    if value["gamma"] != 1:
        rgb = rgb.point([round(255 * (i / 255) ** (1 / value["gamma"])) for i in range(256)] * 3)
    if value["temperature"]:
        t = value["temperature"] / 100 * 0.25  # +: warmer (more red, less blue)
        lut = ([min(255, round(i * (1 + t))) for i in range(256)] + list(range(256))
               + [min(255, round(i * (1 - t))) for i in range(256)])
        rgb = rgb.point(lut)
    if value["sharpness"] != 1:
        rgb = ImageEnhance.Sharpness(rgb).enhance(value["sharpness"])
    if alpha is not None:
        rgb.putalpha(alpha)
    rgb.info["trim_bottom"] = trim
    return rgb


def auto_enhance(path: Path, params: dict) -> dict:
    """Ask the model for slider values; it sees the current crop/stretch with neutral color settings."""
    geometry = {key: params[key] for key in ("crop", "size", "orient") if key in params}
    img = edit_image(path, geometry, REF_SIZE).convert("RGB")
    luma = img.convert("L")
    hist = luma.histogram()
    total = sum(hist)

    def percentile(q: float) -> int:
        count = 0
        for level, n in enumerate(hist):
            count += n
            if count >= total * q:
                return level
        return 255

    r, g, b = ImageStat.Stat(img).mean
    stats = {"mean": round(ImageStat.Stat(luma).mean[0]), "p2": percentile(.02), "p50": percentile(.5),
             "p98": percentile(.98), "mean_saturation": round(ImageStat.Stat(img.convert("HSV")).mean[1]),
             "red_blue_ratio": round(r / max(b, 1), 2)}
    job_dir = ROOT / ".yunaviewer" / "jobs" / uuid.uuid4().hex
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        img.save(job_dir / "photo.jpg", "JPEG", quality=90)
        response = call_agent(job_dir, {"mode": "enhance", "model": SEARCH_MODEL, "references": ["photo.jpg"],
                                        "stats": stats})
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    values = {key: min(max(float(response.get(key, 0 if key == "temperature" else 1)), lo), hi)
              for key, (lo, hi) in EDIT_RANGES.items()}
    return {**values, "message": response.get("message", "")}


def edit_command(prompt: str, controls: dict, body: list, path: Path | None = None, params: dict | None = None) -> dict:
    """Natural-language editor command -> new control values (text only, no image sent).
    Body edits come back as per-part totals and are merged into the accumulated list here.
    Face shape edits are dropped with an explanation when the photo's face is too small for them."""
    job_dir = ROOT / ".yunaviewer" / "jobs" / uuid.uuid4().hex
    try:
        response = call_agent(job_dir, {"mode": "edit", "model": SEARCH_MODEL, "references": [],
                                        "prompt": prompt, "state": controls, "body": body})
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    if path is not None and any(response.get(key) is not None for key in effects.SHAPE_KEYS):
        rot, flip = orient_params(params or {})
        body_json = json.dumps((params or {}).get("body") or [], sort_keys=True)
        key = (str(path), path.stat().st_mtime_ns, rot, flip)
        height = (body_warped(*key, body_json) if body_json != "[]" else load_base(str(path), rot, flip)).height
        reason = effects.face_too_small(face_points(*key, body_json), height)
        if reason:
            for name in effects.SHAPE_KEYS:
                response.pop(name, None)
            response["message"] = reason
    if response.get("body"):
        response["body"] = merge_body(body, response["body"])
    elif response.get("reset"):
        response["body"] = []
    else:
        response.pop("body", None)
    return response


def has_transparency(img: Image.Image) -> bool:
    return img.mode == "RGBA" and img.getchannel("A").getextrema()[0] < 255


def checkerboard(img: Image.Image, cell: int = 12) -> Image.Image:
    """Show transparency in the preview: RGBA image over a gray checkerboard."""
    import numpy as np

    h, w = img.height, img.width
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // cell + xx // cell) % 2)[..., None] == 0, 200, 150).astype(np.uint8).repeat(3, axis=2)
    out = Image.fromarray(board)
    out.paste(img.convert("RGB"), mask=img.getchannel("A"))
    return out


def save_edited(path: Path, params: dict) -> Path:
    img = edit_image(path, params)
    suffix = path.suffix.lower()
    if has_transparency(img) and suffix != ".png":
        suffix = ".png"  # background removal needs an alpha channel
    # editing an edited copy (x_edit.png) saves the next version of the original (x_edit2.png), not x_edit_edit.png,
    # so every version stays grouped under the original
    stem = library.EDIT_SUFFIX_RE.sub("", path.name).rsplit(".", 1)[0]
    for n in range(1, 1000):
        target = path.with_name(f"{stem}_edit{'' if n == 1 else n}{suffix}")
        if not target.exists():
            break
    if suffix in (".jpg", ".jpeg"):
        img.convert("RGB").save(target, "JPEG", quality=95)
    else:
        img.save(target)
    return target


# --- albums, duplicates, myshare-style renaming ---

def album_groups(rel_dir: str, kind: str) -> dict:
    paths = folder_images(safe_path(rel_dir))
    tags = stored_tags(paths)
    entries = [(p.name, search_tags(p, tags[p])) for p in paths if p in tags]
    return {"groups": library.build_albums(kind, entries), "untagged": len(paths) - len(entries)}


def folder_tag_map(rel_dir: str) -> dict:
    paths = folder_images(safe_path(rel_dir))
    return {path.name: tags for path, tags in stored_tags(paths).items()}


def best_shot(paths: list[Path]) -> dict:
    """Score similar photos locally (no upload) and name the best one: sharp face, open eyes, sane exposure,
    the bigger file when the rest is equal."""
    import numpy as np
    import body_edit

    if not 2 <= len(paths) <= 30:
        raise ValueError("베스트 컷은 비슷한 사진 2~30장 중에서 고를 수 있습니다.")
    measures = []
    for path in paths:
        with Image.open(path) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
        pixels = img.width * img.height
        img.thumbnail((1024, 1024))  # enough for the face model and sharpness, fast on 4k files
        with body_edit.LOCK:
            face = body_edit.detect_face(np.ascontiguousarray(np.array(img)))
        measures.append({"name": path.name, "pixels": pixels, **effects.shot_measures(img, face)})
    ranked = effects.rank_shots(measures)
    best = max(ranked, key=lambda m: (m["score"], m["pixels"], -len(m["name"])))
    return {"best": best["name"], "shots": [{k: m[k] for k in ("name", "score", "reasons")} for m in ranked]}


def duplicate_groups(rel_dir: str) -> dict:
    paths = folder_images(safe_path(rel_dir))
    return {"groups": library.find_duplicates(ROOT, paths, thumbnail, file_hash)}


def rename_plan(rel_dir: str, names: list[str], allow_index: bool = False) -> dict:
    """Propose myshare-style names for the given files (default: every file of the folder not yet in that form)."""
    directory = safe_path(rel_dir)
    images = {p.name: p for p in folder_images(directory)}
    targets = [images[n] for n in names if n in images] if names else [
        p for n, p in images.items() if library.needs_name(n)]
    if not targets:
        raise ValueError("이름을 정리할 이미지가 없습니다. (선택한 이미지가 없고 정리 안 된 파일도 없습니다)")
    ensure_described(targets, directory, allow_index)
    tags = stored_tags(targets)
    entries = [(p.name, tags[p]) for p in targets if p in tags]
    skipped = [p.name for p in targets if p not in tags]
    if not entries:
        raise ValueError(tag_status(directory)["error"] or "이미지 분석이 끝나지 않았습니다. 잠시 후 다시 시도하세요.")

    all_tags = stored_tags(list(images.values()))
    known = {}
    for path in images.values():
        match = library.FINAL_NAME_RE.fullmatch(path.name)
        if match and path in all_tags:
            known.setdefault(match.group("outfit").lower(), library.outfit_description(all_tags[path]))
    clusters = library.cluster_outfits(entries)
    job_dir = ROOT / ".yunaviewer" / "jobs" / uuid.uuid4().hex
    try:
        response = call_agent(job_dir, {
            "mode": "outfit_token", "model": SEARCH_MODEL, "references": [],
            "clusters": [{"id": f"C{i}", "description": library.outfit_description(c[0][1]), "count": len(c)}
                         for i, c in enumerate(clusters, 1)],
            "known": [{"token": token, "description": description} for token, description in known.items()]})
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    tokens = {str(t.get("id")): library.outfit_token(t.get("token", "")) for t in response.get("tokens", [])}
    items = []
    for i, cluster in enumerate(clusters, 1):
        token = tokens.get(f"C{i}") or library.outfit_token(library.outfit_description(cluster[0][1]))[:24]
        items += [{"name": name, "outfit": token, "shot": t.get("visible")} for name, t in cluster]
    items.sort(key=lambda item: item["name"])
    plan = library.plan_names(directory, items)
    planned = {item["name"] for item in plan}
    return {"plan": plan, "skipped": skipped + [e[0] for e in entries if e[0] not in planned]}


def rename_apply(rel_dir: str, plan: list[dict]) -> dict:
    directory = safe_path(rel_dir)
    for item in plan:
        for key in ("name", "new_name"):
            if not isinstance(item.get(key), str) or Path(item[key]).name != item[key] or item[key].startswith("."):
                raise ValueError("이름 형식이 올바르지 않습니다.")
        if Path(item["new_name"]).suffix.lower() not in IMAGE_EXTS:
            raise ValueError("이미지 확장자가 필요합니다.")
    if not plan:
        raise ValueError("바꿀 이름이 없습니다.")
    return {"op": library.apply_rename(ROOT, directory, plan)["id"], "count": len(plan)}


# --- PWA: manifest, service worker and generated icons (installable on a phone's home screen) ---

MANIFEST = json.dumps({
    "name": "Yuna Viewer", "short_name": "Yuna", "start_url": "/", "display": "standalone",
    "background_color": "#111111", "theme_color": "#1b1b1b",
    "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
              {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}]})

# Offline browsing of what was seen: the app page and thumbnails are cached, folder listings are network-first.
SERVICE_WORKER = """const CACHE = 'yunaviewer-v1';
self.addEventListener('install', e => { self.skipWaiting(); });
self.addEventListener('activate', e => e.waitUntil(
  caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));
async function networkFirst(req) {
  const cache = await caches.open(CACHE);
  try { const res = await fetch(req); if (res.ok) cache.put(req, res.clone()); return res; }
  catch (error) { const hit = await cache.match(req); if (hit) return hit; throw error; }
}
async function cacheFirst(req) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req);
  if (hit) { fetch(req).then(res => res.ok && cache.put(req, res)).catch(() => {}); return hit; }
  const res = await fetch(req); if (res.ok) cache.put(req, res.clone()); return res;
}
self.addEventListener('fetch', e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return;
  if (url.pathname === '/thumb') e.respondWith(cacheFirst(req));
  else if (url.pathname === '/' || url.pathname === '/api/list' || url.pathname.startsWith('/icon-')) e.respondWith(networkFirst(req));
});
"""


@functools.lru_cache(maxsize=2)
def app_icon(size: int) -> bytes:
    """Rounded gradient square with a Y, drawn at 4x and scaled down for smooth edges."""
    import numpy as np

    big = size * 4
    yy, xx = np.mgrid[0:big, 0:big]
    t = ((xx + yy) / (2 * (big - 1)))[..., None]  # diagonal blue -> pink
    start, stop = np.array((77, 156, 255)), np.array((255, 122, 168))
    gradient = Image.fromarray((start + (stop - start) * t).astype(np.uint8)).convert("RGBA")
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, big - 1, big - 1), radius=big // 5, fill=255)
    gradient.putalpha(mask)
    letter = ImageDraw.Draw(gradient)
    letter.text((big // 2, big // 2), "Y", fill="white", anchor="mm", font=ImageFont.load_default(size=int(big * 0.6)))
    buffer = io.BytesIO()
    gradient.resize((size, size), Image.Resampling.LANCZOS).save(buffer, "PNG")
    return buffer.getvalue()


class Handler(http.server.BaseHTTPRequestHandler):
    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data: bytes, content_type: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        # images are cached; index.html is always revalidated so UI edits show up on reload
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-cache")
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        rel = query.get("path", query.get("dir", [""]))[0]
        try:
            if url.path == "/":
                self.send_file(INDEX_HTML)
            elif url.path == "/api/list":
                listing = list_dir(rel)
                self.send_json(listing)
            elif url.path == "/api/dirs":
                self.send_json({"dirs": library.list_dirs(ROOT)})
            elif url.path == "/api/albums":
                self.send_json(album_groups(rel, query.get("kind", ["outfit"])[0]))
            elif url.path == "/api/duplicates":
                self.send_json(duplicate_groups(rel))
            elif url.path == "/api/metadata":
                self.send_json(image_metadata(rel))
            elif url.path == "/api/tags/list":
                self.send_json({"tags": folder_tag_map(rel)})
            elif url.path == "/api/edit/base":
                path = safe_path(rel)
                if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                    raise FileNotFoundError(rel)
                rot, flip = orient_params({"orient": {"rot": int(query.get("rot", ["0"])[0]),
                                                      "flip": query.get("flip", ["0"])[0] == "1"}})
                base = load_base(str(path), rot, flip).convert("RGB")
                base.thumbnail((2400, 2400))
                buffer = io.BytesIO()
                base.save(buffer, "JPEG", quality=90)
                self.send_bytes(buffer.getvalue(), "image/jpeg")
            elif url.path == "/manifest.webmanifest":
                self.send_bytes(MANIFEST.encode(), "application/manifest+json")
            elif url.path == "/sw.js":
                self.send_bytes(SERVICE_WORKER.encode(), "text/javascript")
            elif url.path in ("/icon-192.png", "/icon-512.png"):
                self.send_bytes(app_icon(int(url.path[6:9])), "image/png", cache=True)
            elif url.path == "/api/tags/status":
                self.send_json(tag_status_snapshot(safe_path(rel)))
            elif url.path in ("/thumb", "/img"):
                path = safe_path(rel)
                if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                    raise FileNotFoundError(rel)
                self.send_file(thumbnail(path) if url.path == "/thumb" else path, cache=True)
            else:
                self.send_error(404)
        except PermissionError:
            self.send_error(403)
        except (FileNotFoundError, NotADirectoryError):
            self.send_error(404)
        except (ValueError, TypeError) as error:  # e.g. rot=abc, kind=nonsense
            self.send_json({"error": str(error)}, 400)

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/api/agent":
                prompt = str(body.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("프롬프트를 입력하세요.")
                selected = list(body.get("selected", []))
                if is_question(prompt, selected):
                    self.send_json(ask_images(str(body.get("dir", "")), prompt, selected))
                else:
                    within = body.get("within")
                    self.send_json(run_search(str(body.get("dir", "")), prompt, selected, bool(body.get("allow_index")),
                                              [str(n) for n in within] if isinstance(within, list) else None,
                                              str(body.get("previous", ""))[:300]))
            elif self.path == "/api/edit/command":
                prompt = str(body.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("프롬프트를 입력하세요.")
                photo = safe_path(str(body["path"])) if body.get("path") else None
                if photo is not None and (photo.suffix.lower() not in IMAGE_EXTS or not photo.is_file()):
                    raise FileNotFoundError(str(body["path"]))
                self.send_json(edit_command(prompt, dict(body.get("controls") or {}), list(body.get("body") or []),
                                            photo, dict(body.get("params") or {})))
            elif self.path == "/api/tags/start":
                directory = safe_path(str(body.get("dir", "")))
                if not directory.is_dir() or library.is_state_path(ROOT, directory):
                    raise ValueError("AI 색인을 만들 수 없는 폴더입니다.")
                if not folder_images(directory):
                    raise ValueError("색인할 이미지가 없습니다.")
                start_tagging(directory)
                self.send_json(tag_status_snapshot(directory))
            elif self.path == "/api/fav":
                paths = [safe_path(r) for r in body.get("paths", [])]
                self.send_json({"count": library.set_favorites(ROOT, paths, bool(body.get("value", True)))})
            elif self.path == "/api/marks":
                paths = [safe_path(r) for r in body.get("paths", [])]
                if not paths or any(not p.is_file() or p.suffix.lower() not in IMAGE_EXTS for p in paths):
                    raise ValueError("이미지를 선택하세요.")
                rating = int(body["rating"]) if body.get("rating") is not None else None
                note = str(body["note"]) if body.get("note") is not None else None
                self.send_json({"count": library.set_marks(ROOT, paths, rating, note)})
            elif self.path == "/api/bestshot":
                paths = [safe_path(r) for r in body.get("paths", [])]
                if any(not p.is_file() or p.suffix.lower() not in IMAGE_EXTS for p in paths):
                    raise ValueError("이미지가 아닌 항목이 있습니다.")
                self.send_json(best_shot(paths))
            elif self.path == "/api/trash/empty":  # paths: delete only those, otherwise everything
                paths = [safe_path(r) for r in body["paths"]] if "paths" in body else None
                if paths == []:
                    raise ValueError("선택한 이미지가 없습니다.")
                self.send_json({"count": library.empty_trash(ROOT, paths)})
            elif self.path in ("/api/files/trash", "/api/files/move", "/api/files/restore"):
                paths = [safe_path(r) for r in body.get("paths", [])]
                if not paths:
                    raise ValueError("선택한 이미지가 없습니다.")
                if self.path == "/api/files/trash":
                    op = library.trash_files(ROOT, paths)
                elif self.path == "/api/files/restore":
                    self.send_json({"count": library.restore_from_trash(ROOT, paths)})
                    return
                else:
                    dest = safe_path(str(body.get("dest", "")))
                    if body.get("new_folder"):
                        dest = library.make_dir(ROOT, dest, str(body["new_folder"]))
                    op = library.move_files(ROOT, paths, dest)
                self.send_json({"op": op["id"], "count": len(op["items"])})
            elif self.path == "/api/files/undo":
                self.send_json(library.undo_op(ROOT, str(body.get("id", ""))))
            elif self.path == "/api/rename/plan":
                self.send_json(rename_plan(str(body.get("dir", "")), [str(n) for n in body.get("names", [])],
                                           bool(body.get("allow_index"))))
            elif self.path == "/api/rename/apply":
                self.send_json(rename_apply(str(body.get("dir", "")), list(body.get("plan", []))))
            elif self.path in ("/api/edit/preview", "/api/edit/save", "/api/edit/auto"):
                path = safe_path(str(body.get("path", "")))
                if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                    raise ValueError("이미지 파일이 아닙니다.")
                params = dict(body.get("params") or {})
                if self.path == "/api/edit/save":
                    self.send_json({"name": save_edited(path, params).name})
                elif self.path == "/api/edit/auto":
                    self.send_json(auto_enhance(path, params))
                else:
                    buffer = io.BytesIO()
                    preview = edit_image(path, params, PREVIEW_SIDE)
                    (checkerboard(preview) if has_transparency(preview) else preview.convert("RGB")).save(
                        buffer, "JPEG", quality=88)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("X-Trim-Bottom", str(preview.info.get("trim_bottom", 0)))
                    self.send_header("Content-Length", str(buffer.tell()))
                    self.end_headers()
                    self.wfile.write(buffer.getvalue())
            else:
                self.send_error(404)
        except PermissionError:
            self.send_json({"error": "허용되지 않은 경로입니다."}, 403)
        except NeedsIndex as error:  # 409: nothing was sent anywhere, the client may ask and retry with allow_index
            self.send_json({"error": str(error), "needs_index": error.count}, 409)
        except (ValueError, TypeError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            self.send_json({"error": str(error)}, 400)


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=CONFIG.get("server", "root", fallback=str(ROOT)))
    parser.add_argument("--port", type=int, default=CONFIG.getint("server", "port", fallback=8890))
    parser.add_argument("--host", default=CONFIG.get("server", "host", fallback="127.0.0.1"))
    args = parser.parse_args()
    ROOT = Path(args.root).expanduser().resolve()
    print(f"yunaviewer: {ROOT} -> http://localhost:{args.port}")
    http.server.ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
