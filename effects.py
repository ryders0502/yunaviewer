"""Image effects that need no model of their own: orientation, background blur/removal, face-only tone.

The person mask and face landmarks come from body_edit.detect(); everything here is plain PIL/OpenCV and
tested on synthetic images.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from body_edit import FACE_OVAL

# MediaPipe FaceMesh index groups (same lists as yunareview): kept out of skin smoothing
FACE_EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
             263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
FACE_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 191, 80, 81, 82,
              13, 312, 311, 310, 415, 308]

BG_MODES = ("none", "blur", "remove")


def orient(img: Image.Image, rot: int = 0, flip: bool = False) -> Image.Image:
    """Mirror horizontally first (if flip), then rotate clockwise by rot (0/90/180/270)."""
    if flip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    turn = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}.get(rot % 360)
    return img.transpose(turn) if turn is not None else img


def _clean_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Soft person mask -> crisper, slightly feathered alpha at the image size (w, h)."""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape[:2] != (size[1], size[0]):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    mask = np.clip((mask - 0.3) / 0.4, 0.0, 1.0)  # sharpen the soft edge of the model output
    sigma = max(1.0, min(size) / 500)
    return cv2.GaussianBlur(mask, (0, 0), sigma)


def apply_background(img: Image.Image, mask: np.ndarray, mode: str, amount: float = 0.5) -> Image.Image:
    """blur: keep the person sharp and blur the rest; remove: make the background transparent (RGBA)."""
    if mode not in BG_MODES or mode == "none":
        return img
    alpha = _clean_mask(mask, img.size)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    if mode == "remove":
        out = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
        if img.mode == "RGBA":  # keep an existing transparency too
            alpha = alpha * (np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0)
        out.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
        return out
    sigma = max(0.5, float(amount) * 0.04 * max(img.size))
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigma)
    mixed = rgb * alpha[..., None] + blurred * (1 - alpha[..., None])
    out = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))
    if img.mode == "RGBA":
        out.putalpha(img.getchannel("A"))
    return out


def _hull_mask(points: np.ndarray, shape: tuple[int, int], dilate: float = 0.0) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(points[:, :2]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate > 0:
        k = max(1, int(dilate)) * 2 + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return mask.astype(np.float32) / 255.0


def face_mask(face_px: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Feathered mask of the face oval and the face width in pixels."""
    oval = face_px[FACE_OVAL]
    width = float(oval[:, 0].max() - oval[:, 0].min())
    mask = _hull_mask(oval, shape)
    sigma = max(1.0, width * 0.04)
    return cv2.GaussianBlur(mask, (0, 0), sigma), width


def adjust_face(img: Image.Image, face_px: np.ndarray, brightness: float = 1.0, tone: float = 0.0,
                smooth: float = 0.0) -> Image.Image:
    """Face-only brightness (multiplier), warm/cool tone (-100..100) and skin smoothing (0..1)."""
    if brightness == 1 and tone == 0 and smooth == 0:
        return img
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    mask, face_width = face_mask(np.asarray(face_px, dtype=np.float64), (h, w))
    result = rgb.copy()
    if smooth > 0:
        # bilateral filter on the face region only (it is slow on whole images); eyes and mouth keep their detail
        ys, xs = np.nonzero(mask > 0.02)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            crop = rgb[y0:y1, x0:x1].astype(np.uint8)
            d = max(5, int(face_width * 0.04) | 1)
            soft = cv2.bilateralFilter(crop, d, 28 + 30 * smooth, d * 2).astype(np.float32)
            keep = np.maximum(_hull_mask(np.asarray(face_px, dtype=np.float64)[FACE_EYES], (h, w), face_width * 0.03),
                              _hull_mask(np.asarray(face_px, dtype=np.float64)[FACE_MOUTH], (h, w), face_width * 0.02))
            keep = cv2.GaussianBlur(keep, (0, 0), max(1.0, face_width * 0.02))[y0:y1, x0:x1]
            weight = (mask[y0:y1, x0:x1] * (1 - keep) * min(1.0, smooth))[..., None]
            result[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1 - weight) + soft * weight
    adjusted = result * float(brightness)
    if tone:
        t = float(tone) / 100 * 0.2
        adjusted[..., 0] *= 1 + t
        adjusted[..., 2] *= 1 - t
    m = mask[..., None]
    mixed = np.where(brightness == 1 and tone == 0, result, result * (1 - m) + np.clip(adjusted, 0, 255) * m)
    out = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))
    if img.mode == "RGBA":
        out.putalpha(img.getchannel("A"))
    return out


# --- face shape: nose / eye size, lip thickness, jaw width (local warps from the 478 face landmarks) ---

SHAPE_KEYS = ("nose_size", "eye_size", "upper_lip", "lower_lip", "jaw_width")
MAX_SHAPE_PCT = 20.0
MIN_FACE_PX, MIN_FACE_RATIO = 200, 0.12  # smaller faces (full-body shots) give no visible, clean result
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def face_too_small(face_px: np.ndarray | None, image_height: int) -> str:
    """'' when the face is large enough for shape edits, otherwise the reason (Korean, shown to the user)."""
    if face_px is None:
        return "얼굴이 작거나 인식되지 않아 코·눈·입술·턱선 보정을 할 수 없어요. 얼굴이 크게 나온 사진에서 사용하세요."
    ys = np.asarray(face_px)[FACE_OVAL, 1]
    height = ys.max() - ys.min()
    if height < MIN_FACE_PX or height < MIN_FACE_RATIO * image_height:
        return "얼굴이 작게 나온 사진이라 코·눈·입술·턱선 보정을 할 수 없어요. 얼굴이 크게 나온 사진에서 사용하세요."
    return ""


def _falloff(x: np.ndarray) -> np.ndarray:
    """1 at x=0, smoothly 0 at x>=1."""
    return np.clip(1 - x * x, 0, 1) ** 2


def _smooth(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def reshape_face(img: Image.Image, face_px: np.ndarray, shape: dict) -> Image.Image:
    """shape: {key: pct} with keys from SHAPE_KEYS, -20..20 (+ bigger/thicker/wider). Only a box around the
    face is remapped. Works in a face frame (u along the eye line, v toward the chin), so tilted heads work."""
    shape = {k: max(-MAX_SHAPE_PCT, min(MAX_SHAPE_PCT, float(v))) for k, v in shape.items() if k in SHAPE_KEYS and v}
    if not shape:
        return img
    pts = np.asarray(face_px, dtype=np.float64)[:, :2]
    left, right = pts[LEFT_EYE].mean(0), pts[RIGHT_EYE].mean(0)
    ex = (right - left) / np.linalg.norm(right - left)
    ey = np.array([-ex[1], ex[0]])
    if np.dot(pts[152] - (left + right) / 2, ey) < 0:  # v must point toward the chin
        ey = -ey
    origin = pts[FACE_OVAL].mean(0)
    uv = (pts - origin) @ np.stack([ex, ey], 1)  # landmark coords in the face frame
    face_h = np.ptp(uv[FACE_OVAL, 1])

    arr = np.asarray(img)
    h, w = arr.shape[:2]
    lo, hi = pts[FACE_OVAL].min(0) - face_h * 0.4, pts[FACE_OVAL].max(0) + face_h * 0.4
    x0, y0 = max(0, int(lo[0])), max(0, int(lo[1]))
    x1, y1 = min(w, int(hi[0]) + 1), min(h, int(hi[1]) + 1)
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    rel = np.stack([gx - origin[0], gy - origin[1]], -1)
    u, v = rel @ ex, rel @ ey
    du, dv = np.zeros_like(u), np.zeros_like(v)
    k = {key: 1 / (1 + pct / 100) - 1 for key, pct in shape.items()}  # inverse map: >0 shrinks

    def radial(center, radius, amount):  # scale everything within radius around center
        f = amount * _falloff(np.hypot(u - center[0], v - center[1]) / radius)
        du[:] += (u - center[0]) * f
        dv[:] += (v - center[1]) * f

    if "nose_size" in k:
        radial(uv[[1, 2, 98, 327]].mean(0), np.linalg.norm(uv[129] - uv[358]), k["nose_size"])
    if "eye_size" in k:
        for eye, (a, b) in ((LEFT_EYE, (33, 133)), (RIGHT_EYE, (362, 263))):
            radial(uv[eye].mean(0), np.linalg.norm(uv[a] - uv[b]), k["eye_size"])
    mouth_u, mouth_half = (uv[61, 0] + uv[291, 0]) / 2, abs(uv[291, 0] - uv[61, 0]) / 2
    across = _falloff(np.abs(u - mouth_u) / (mouth_half * 1.15))
    for key, inner, outer in (("lower_lip", 14, 17), ("upper_lip", 13, 0)):
        if key not in k:
            continue
        thick = uv[outer, 1] - uv[inner, 1]  # signed: + below the mouth, - above it
        a = (v - uv[inner, 1]) / thick  # 0 at the lip line, 1 at the lip's outer edge
        weight = np.where(a > 0, 1 - _smooth(1.0, 2.5, a), 0) * across
        dv += (v - uv[inner, 1]) * k[key] * weight
    if "jaw_width" in k:
        center_u, half = (uv[234, 0] + uv[454, 0]) / 2, abs(uv[454, 0] - uv[234, 0]) / 2
        chin = uv[152, 1]
        eyes = uv[LEFT_EYE + RIGHT_EYE, 1].mean()
        jaw = max(uv[[172, 397], 1].mean(), eyes + face_h * 0.2)  # jaw corners well below the eyes
        # starts at eye level, full at the jaw corners: a long ramp keeps the cheek outline from bending
        down = _smooth(eyes, jaw, v) * (1 - _smooth(chin, chin + face_h * 0.25, v))
        side = 1 - _smooth(0.9, 2.0, np.abs(u - center_u) / half)  # a wide fade keeps hair beside the jaw from folding
        du += (u - center_u) * k["jaw_width"] * down * side

    src = np.stack([gx, gy], -1) + du[..., None] * ex + dv[..., None] * ey
    out = arr.copy()
    out[y0:y1, x0:x1] = cv2.remap(arr, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32),
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(out, img.mode)


# --- best shot among similar photos: sharpness, open eyes, exposure, resolution (all local) ---

def eye_openness(face_px: np.ndarray) -> float:
    """Eye aspect ratio averaged over both eyes: about 0.25-0.35 open, under 0.15 closed."""
    pts = np.asarray(face_px, dtype=np.float64)[:, :2]
    ratio = [np.linalg.norm(pts[top] - pts[bottom]) / max(1e-6, np.linalg.norm(pts[a] - pts[b]))
             for top, bottom, a, b in ((159, 145, 33, 133), (386, 374, 362, 263))]
    return float(np.mean(ratio))


def shot_measures(img: Image.Image, face_px: np.ndarray | None) -> dict:
    """Raw measures of one photo; compare them only within a group of similar photos."""
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    region = gray
    if face_px is not None:  # sharpness where it matters: the face
        oval = np.asarray(face_px)[FACE_OVAL, :2]
        (x0, y0), (x1, y1) = np.maximum(oval.min(0), 0).astype(int), oval.max(0).astype(int)
        if x1 - x0 > 16 and y1 - y0 > 16:
            region = gray[y0:y1, x0:x1]
    return {"sharpness": float(cv2.Laplacian(region, cv2.CV_64F).var()),
            "eyes": eye_openness(face_px) if face_px is not None else None,
            "clipped": float(((gray <= 3) | (gray >= 252)).mean())}


def rank_shots(measures: list[dict]) -> list[dict]:
    """measures: [{sharpness, eyes, clipped, pixels}] of similar photos -> same list with score (0..1) and
    short Korean reasons. Sharpness and resolution are relative to the best in the group."""
    sharpest = max(m["sharpness"] for m in measures) or 1.0
    largest = max(m["pixels"] for m in measures) or 1
    has_eyes = any(m["eyes"] is not None for m in measures)
    ranked = []
    for m in measures:
        sharp, size = m["sharpness"] / sharpest, m["pixels"] / largest
        eyes = 1.0 if m["eyes"] is None else min(1.0, m["eyes"] / 0.2)
        exposure = 1 - min(1.0, m["clipped"] * 10)
        score = 0.5 * sharp + (0.25 * eyes if has_eyes else 0) + 0.15 * exposure + 0.1 * size
        score /= 0.75 + (0.25 if has_eyes else 0)
        reasons = []
        if sharp >= 0.999:
            reasons.append("가장 선명")
        elif sharp < 0.6:
            reasons.append("흐림")
        if m["eyes"] is not None and m["eyes"] < 0.15:
            reasons.append("눈 감김")
        if m["clipped"] > 0.05:
            reasons.append("노출 과다/부족")
        if size >= 0.999 and len(measures) > 1 and min(x["pixels"] for x in measures) < largest:
            reasons.append("해상도 가장 큼")
        ranked.append({**m, "score": round(score, 3), "reasons": reasons})
    return ranked


# --- spot removal: painted strokes are filled in from their surroundings (OpenCV inpainting, local) ---

MAX_STROKES, MAX_POINTS = 300, 4000


def erase_mask(size: tuple[int, int], strokes: list[dict]) -> np.ndarray:
    """uint8 mask (h, w) of the painted strokes: [{r: radius px, pts: [[x, y], ...]}] in image pixels."""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    for stroke in strokes[:MAX_STROKES]:
        r = int(round(min(max(float(stroke.get("r", 0)), 1), 500)))
        pts = np.round(np.asarray(stroke.get("pts") or [], dtype=np.float64)[:MAX_POINTS]).astype(np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2 or not len(pts):
            continue
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, thickness=2 * r, lineType=cv2.LINE_AA)
        for x, y in pts[[0, -1]]:
            cv2.circle(mask, (int(x), int(y)), r, 255, -1, cv2.LINE_AA)
    return mask


def erase_spots(img: Image.Image, strokes: list[dict]) -> Image.Image:
    """Fill the painted areas from what is around them. Only a box around the strokes is processed, so small
    spots on 4k photos stay fast. Works best on small areas; large ones come out smeared."""
    if not strokes:
        return img
    mask = erase_mask(img.size, strokes)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return img
    rgb = np.asarray(img.convert("RGB")).copy()
    pad = 24 + int(max(float(s.get("r", 0)) for s in strokes[:MAX_STROKES]) * 2)
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(rgb.shape[1], xs.max() + pad + 1), min(rgb.shape[0], ys.max() + pad + 1)
    roi = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    # the mask edge is anti-aliased: everything touched counts, so no half-painted ring is left behind
    hole = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    rgb[y0:y1, x0:x1] = cv2.inpaint(roi, hole, 5, cv2.INPAINT_TELEA)
    out = Image.fromarray(rgb)
    if img.mode == "RGBA":
        out.putalpha(img.getchannel("A"))
    return out
