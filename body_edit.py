"""Body reshaping (limb length/thickness, chest/waist/pelvis width, head, whole body).

Copied from ~/workspace/yunareview/app.py (pose + segmentation via MediaPipe, MLS affine warp).
Only the editing path is kept; measurement/comparison code is left out. edit_image_array() is the
entry point; detect() replaces yunareview's extract_landmarks() and returns only what editing needs.
"""
import os
import threading
import urllib.request

import cv2
import numpy as np
from PIL import Image as PILImage

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL, L_FOOT, R_FOOT = 29, 30, 31, 32

# MediaPipe FaceMesh face oval: fixed anchors so head edits do not distort the face
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]
FACE_GROUPS = {"oval": FACE_OVAL}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class DetectionError(Exception):
    pass


def _face_landmarks_to_px(face_lm_norm, img_w, img_h, offset_x=0.0, offset_y=0.0):
    """정규화(0~1) 얼굴 랜드마크 (N,3) -> 픽셀 좌표. offset_x/y를 주면 그만큼 더해서
    원본 전체 이미지 기준 절대 좌표로 만든다(크롭에서 검출됐을 때 크롭의 좌상단 오프셋).

    얼굴 검출은 전체 이미지에서 실패하면 머리 부분만 정사각형으로 크롭해 재시도한다
    (_head_crop_box). 크롭은 리사이즈 없이 픽셀 그대로 잘라낸 거라 원본과 같은 px 단위를
    공유하지만, 정규화(0~1) 좌표는 "그 프레임 자신의 폭/높이"로 나눈 값이라 원본(임의
    종횡비)과 정사각형 크롭은 x/y 스케일 비율이 서로 다르다 — 코드 리뷰로 지적된 버그: 표본은
    전체 이미지에서, 비교는 크롭에서 얼굴을 검출했다면 정규화 좌표를 그대로 비교하는 순간
    얼굴 형태(Procrustes)·크기 비교에 종횡비 차이만큼 인위적 왜곡이 생긴다. 픽셀로 변환하면
    어느 프레임에서 검출됐든 실제 종횡비가 살아있어 이 왜곡이 사라진다.

    offset_x/y는 Procrustes·bbox 크기 비교에는 영향 없다(둘 다 평행이동 불변) — 하지만 얼굴
    랜드마크를 포즈 랜드마크(pose_px)와 같은 좌표계에서 같이 써야 하는 곳(예: /edit 워프의
    얼굴 보호용 앵커점)에는 절대 위치가 맞아야 해서 필요하다."""
    arr = np.asarray(face_lm_norm, dtype=float)
    return np.column_stack([arr[:, 0] * img_w + offset_x, arr[:, 1] * img_h + offset_y, arr[:, 2] * img_w])


def _mask_row_center_bounds(mask_row, center_x, threshold=0.5):
    """mask_row: 1D 배열(한 행). center_x: 그 행에서 몸통 중심으로 볼 x좌표(픽셀).
    center_x를 포함하는 연속 구간만 좌우로 걸어나가며 경계를 찾는다 (left, right) 반환.

    첫/마지막 켜진 픽셀을 그냥 쓰면(예전 방식) 팔이 몸통에서 떨어져 있을 때 그 사이 빈
    공간까지, 또는 마스크의 분리된 잡음 영역까지 폭에 포함되는 문제가 있었다(코드 리뷰 지적,
    실측으로도 재현 가능). center_x 자체가 마스크 바깥이면 None."""
    n = len(mask_row)
    cx = int(round(center_x))
    if cx < 0 or cx >= n or mask_row[cx] <= threshold:
        return None
    left = cx
    while left > 0 and mask_row[left - 1] > threshold:
        left -= 1
    right = cx
    while right < n - 1 and mask_row[right + 1] > threshold:
        right += 1
    return left, right


def _mask_width_perpendicular(mask_arr, p1, p2, threshold=0.5, max_half=None, return_points=False,
                               symmetric_fallback=False):
    """p1-p2 중점에서, p1->p2 방향에 수직으로 실루엣 폭(px)을 잰다. 중점부터 좌우로 걸어나가며
    마스크값이 threshold 아래로 떨어지는 지점까지. 중점 자체가 이미 바깥이면 None.

    return_points=True면 폭(float) 대신 실제 좌우 경계 좌표 (pt_pos, pt_neg)를 반환한다
    (plan_edit()이 이 경계점을 이미지 워프 제어점으로 쓰기 위함).

    팔이 몸통에 딱 붙어있으면(팔을 몸에 붙이고 있는 포즈) 한쪽 방향으로 아무리 걸어도
    몸통 실루엣이 계속 이어져서 max_half(기본 80px) 안에 배경을 못 만날 수 있다. 이 경우
    "80px"이라는 탐색 한계 자체가 측정값으로 둔갑해버려서(실측으로 확인 — 두 이미지 해상도가
    다르면 이 상수 80이 정규화 후 수십 % 오차로 번짐) 기본은 None을 반환해 "측정 불가" 처리한다.

    symmetric_fallback=True면, 양쪽 중 정확히 한쪽만 경계를 못 찾았을 때 그 한쪽을 측정
    불가로 버리는 대신 반대쪽에서 찾은 절반 폭을 좌우 대칭이라 가정하고 그대로 미러링한다.
    다리를 붙이고 선 포즈에서 허벅지 안쪽으로 스캔하면 배경이 아니라 반대쪽 허벅지 실루엣으로
    계속 이어져 그쪽만 못 찾는 경우가 실측으로 확인됨 — 팔은 이 경우가 없어(딱 붙어도 몸통과
    비슷한 두께라 미러링이 부정확) 기본값 False 유지, 허벅지/종아리 두께 호출부만 True로 켠다.
    양쪽 다 못 찾으면(마스크 자체 미검출 등) 여전히 None.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None
    direction = direction / norm
    # 고정 80px는 고해상도 이미지의 굵은 소매/허벅지에서 실제 경계에 닿기 전에 탐색이
    # 끝난다. 뼈대 구간 길이에 비례시키되 작은 이미지에서는 기존 80px를 유지한다.
    max_half = int(max(80, np.ceil(norm * 0.6))) if max_half is None else int(max_half)
    perp = np.array([-direction[1], direction[0]])
    center = (p1 + p2) / 2
    h, w = mask_arr.shape[0], mask_arr.shape[1]

    def mask_at(pt):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= y < h and 0 <= x < w:
            v = mask_arr[y, x]
            return float(v[0]) if hasattr(v, "__len__") else float(v)
        return 0.0

    if mask_at(center) < threshold:
        return None

    pos, pos_found = 0, True
    for step in range(1, max_half):
        if mask_at(center + perp * step) < threshold:
            break
        pos = step
    else:
        pos_found = False  # max_half 안에서 배경을 못 찾음 — 몸통(또는 반대쪽 다리)과 섞여 경계 불명

    neg, neg_found = 0, True
    for step in range(1, max_half):
        if mask_at(center - perp * step) < threshold:
            break
        neg = step
    else:
        neg_found = False

    if not pos_found and not neg_found:
        return None
    if not pos_found or not neg_found:
        if not symmetric_fallback:
            return None
        if pos_found:
            neg = pos
        else:
            pos = neg

    if return_points:
        return center + perp * pos, center - perp * neg
    return float(pos + neg)


def mls_affine_warp(img_arr, src_pts, dst_pts, eps=1e-8, step=4):
    """제어점 src_pts(원본 위치) -> dst_pts(옮길 위치)에 맞춰 이미지 전체를 부드럽게 워프.

    MLS(Moving Least Squares) 아핀 변형 — 제어점 주변은 그 방향으로 국소 확대/축소되고,
    멀어질수록(거리 가중치 1/거리^2) 원본 그대로에 가깝게 감쇠한다. 픽셀마다 가중 최소제곱으로
    국소 2x2 아핀 행렬을 구하는 방식이라(Schaefer et al. MLS deformation의 affine 변형),
    "정확히 pct% 늘림" 같은 국소 스케일 변화를 표현하기에 적합하다(엄밀한 rigid/similarity와
    달리 국소 확대·축소를 허용함 — 이 용도에는 오히려 그게 맞음).

    cv2.remap은 역방향 맵(출력 픽셀마다 원본 어디를 볼지)을 요구하므로, MLS 자체는
    p=dst_pts(그리드가 도는 목표 공간), q=src_pts(그 자리에서 보여줄 원본 좌표)로
    뒤집어서 푼다 — 헷갈리기 쉬운 부분이라 주석으로 명시.
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    h, w = img_arr.shape[:2]
    n = len(src_pts)
    if n == 0:
        return img_arr.copy()

    p, q = dst_pts, src_pts
    # yunaviewer: solve MLS on a coarse grid (every `step` px) and upsample only the displacement,
    # which is smooth; ~step^2 times faster than per pixel with sub-pixel difference.
    full_h, full_w = h, w
    ys, xs = np.mgrid[0:full_h:step, 0:full_w:step]
    h, w = xs.shape
    vx, vy = xs.astype(np.float64), ys.astype(np.float64)  # (h,w) 각각

    # ponytail: n개 제어점을 (n,h,w,..) 로 한 번에 벡터화하면 실사이즈 이미지(수백만 픽셀)에서
    # 수백 MB~수 GB 임시 배열이 여러 개 생겨 메모리 폭발/OOM 킬됨(실측 확인). 제어점 루프로
    # 돌려 매 반복 (h,w) 크기 배열만 쓰도록 함 — n은 많아야 수십 개라 루프 비용은 무시 가능.
    wsum = np.zeros((h, w))
    pstar_x = pstar_y = qstar_x = qstar_y = np.zeros((h, w))
    for i in range(n):
        dx, dy = p[i, 0] - vx, p[i, 1] - vy
        wi = 1.0 / (dx * dx + dy * dy + eps)
        wsum = wsum + wi
        pstar_x = pstar_x + wi * p[i, 0]
        pstar_y = pstar_y + wi * p[i, 1]
        qstar_x = qstar_x + wi * q[i, 0]
        qstar_y = qstar_y + wi * q[i, 1]
    pstar_x, pstar_y = pstar_x / wsum, pstar_y / wsum
    qstar_x, qstar_y = qstar_x / wsum, qstar_y / wsum

    # 픽셀마다: phat @ M ≈ qhat 을 만족하는 2x2 M을 가중 최소제곱으로 구함
    pxx = pxy = pyy = qxx = qxy = qyx = qyy = np.zeros((h, w))
    for i in range(n):
        dx, dy = p[i, 0] - vx, p[i, 1] - vy
        wi = 1.0 / (dx * dx + dy * dy + eps)
        phx, phy = p[i, 0] - pstar_x, p[i, 1] - pstar_y
        qhx, qhy = q[i, 0] - qstar_x, q[i, 1] - qstar_y
        pxx = pxx + wi * phx * phx
        pxy = pxy + wi * phx * phy
        pyy = pyy + wi * phy * phy
        qxx = qxx + wi * phx * qhx
        qxy = qxy + wi * phx * qhy
        qyx = qyx + wi * phy * qhx
        qyy = qyy + wi * phy * qhy

    det = pxx * pyy - pxy * pxy
    det = np.where(np.abs(det) < eps, eps, det)
    inv_xx, inv_xy = pyy / det, -pxy / det
    inv_yx, inv_yy = -pxy / det, pxx / det

    m_xx = inv_xx * qxx + inv_xy * qyx
    m_xy = inv_xx * qxy + inv_xy * qyy
    m_yx = inv_yx * qxx + inv_yy * qyx
    m_yy = inv_yx * qxy + inv_yy * qyy

    vpsx, vpsy = vx - pstar_x, vy - pstar_y
    fx = vpsx * m_xx + vpsy * m_yx + qstar_x
    fy = vpsx * m_xy + vpsy * m_yy + qstar_y

    if step > 1:  # upsample displacement (not the absolute map, which would shift by up to step/2)
        dx = cv2.resize((fx - vx).astype(np.float32), (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize((fy - vy).astype(np.float32), (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        gy, gx = np.mgrid[0:full_h, 0:full_w]
        fx, fy = gx + dx, gy + dy

    return cv2.remap(img_arr, fx.astype(np.float32), fy.astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


LENGTH_CHAINS = {
    "upper": (lambda lm: (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2,
              lambda lm: (lm[L_HIP] + lm[R_HIP]) / 2,
              [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT, R_FOOT]),
    "arm_l": (L_SHOULDER, L_WRIST, [L_WRIST]),
    "arm_r": (R_SHOULDER, R_WRIST, [R_WRIST]),
    "thigh_l": (L_HIP, L_KNEE, [L_KNEE, L_ANKLE, L_HEEL, L_FOOT]),
    "thigh_r": (R_HIP, R_KNEE, [R_KNEE, R_ANKLE, R_HEEL, R_FOOT]),
    "calf_l": (L_KNEE, L_ANKLE, [L_ANKLE, L_HEEL, L_FOOT]),
    "calf_r": (R_KNEE, R_ANKLE, [R_ANKLE, R_HEEL, R_FOOT]),
    "leg_l": (L_HIP, L_ANKLE, [L_ANKLE, L_HEEL, L_FOOT]),
    "leg_r": (R_HIP, R_ANKLE, [R_ANKLE, R_HEEL, R_FOOT]),
}


LIMB_SAMPLE_FRACTIONS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)  # cross-sections along a limb segment

WIDTH_ROW_FRACTIONS = {
    "chest_silhouette": ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), 0.32),
    "pelvis_silhouette": ((L_HIP, R_HIP), (L_KNEE, R_KNEE), 0.15),
}


LIMB_THICKNESS_SEGMENTS = {
    "arm_l_thickness": [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST)],
    "arm_r_thickness": [(R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)],
    "thigh_l_thickness": [(L_HIP, L_KNEE)],
    "thigh_r_thickness": [(R_HIP, R_KNEE)],
    "calf_l_thickness": [(L_KNEE, L_ANKLE)],
    "calf_r_thickness": [(R_KNEE, R_ANKLE)],
}


LEG_THICKNESS_PARTS = {"thigh_l_thickness", "thigh_r_thickness", "calf_l_thickness", "calf_r_thickness"}


WIDTH_PARTS = {"chest_silhouette", "pelvis_silhouette", "waist_silhouette", *LIMB_THICKNESS_SEGMENTS}


EDITABLE_PARTS = {*LENGTH_CHAINS, *WIDTH_PARTS, "head_silhouette", "whole_body"}


MAX_EDIT_PCT = 30.0


def validate_edits(edits):
    """외부 입력 edits를 검증·정규화한다. 반환값은 [{part:str, pct:float}, ...]."""
    if not isinstance(edits, list) or not edits:
        raise ValueError("적용할 보정 명령이 없습니다.")
    if len(edits) > 12:
        raise ValueError("한 번에 적용할 수 있는 보정은 최대 12개입니다.")
    clean = []
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("보정 명령 형식이 올바르지 않습니다.")
        part = edit.get("part")
        if part not in EDITABLE_PARTS:
            raise ValueError(f"지원하지 않는 보정 부위입니다: {part}")
        try:
            pct = float(edit.get("pct"))
        except (TypeError, ValueError):
            raise ValueError(f"{part}: 보정 수치가 올바르지 않습니다.") from None
        if not np.isfinite(pct) or pct == 0 or abs(pct) > MAX_EDIT_PCT:
            raise ValueError(f"{part}: 보정 수치는 0을 제외한 ±{MAX_EDIT_PCT:g}% 범위여야 합니다.")
        clean.append({"part": part, "pct": round(pct, 2)})
    return clean


AGENT_PART_LABELS = {
    "arm_l_thickness": "왼팔 두께", "arm_r_thickness": "오른팔 두께",
    "arm_l": "왼팔 길이", "arm_r": "오른팔 길이",
    "thigh_l_thickness": "왼허벅지 두께", "thigh_r_thickness": "오른허벅지 두께",
    "calf_l_thickness": "왼종아리 두께", "calf_r_thickness": "오른종아리 두께",
    "thigh_l": "왼허벅지 길이", "thigh_r": "오른허벅지 길이",
    "calf_l": "왼종아리 길이", "calf_r": "오른종아리 길이",
    "leg_l": "왼다리 길이", "leg_r": "오른다리 길이",
    "chest_silhouette": "가슴 폭", "pelvis_silhouette": "골반 폭",
    "waist_silhouette": "허리 폭", "head_silhouette": "머리 크기",
    "upper": "상체 길이", "whole_body": "전신 크기",
}


def _mask_row_bounds_px(mask_arr, y_px, center_x_px, threshold=0.5):
    """이미지 y_px(픽셀) 행에서, center_x_px(몸통 중심)를 포함하는 연속 실루엣 구간의
    좌우 경계 x좌표 (left_x, right_x). 실패시 None. measure()용 chest/pelvis/waist_silhouette_
    ratio()와 같은 이유로 몸통 중심 기준 연속 구간만 잰다(_mask_row_center_bounds)."""
    if mask_arr is None:
        return None
    row = int(np.clip(round(y_px), 0, mask_arr.shape[0] - 1))
    mask_row = np.asarray(mask_arr)[row].reshape(-1)
    bounds = _mask_row_center_bounds(mask_row, center_x_px, threshold)
    if bounds is None:
        return None
    return float(bounds[0]), float(bounds[1])


def _width_control_points(pose_px, mask_arr, part, pct):
    """폭 부위(part) pct% 확장/축소에 쓸 (src_pt, dst_pt) 쌍 목록. 실패하면 []."""
    def mid_y(pair):
        return (pose_px[pair[0]][1] + pose_px[pair[1]][1]) / 2

    def mid_x(pair):
        return (pose_px[pair[0]][0] + pose_px[pair[1]][0]) / 2

    scale = 1 + pct / 100.0

    if part in LIMB_THICKNESS_SEGMENTS:
        # yunaviewer: limb thickness no longer goes through MLS (see limb_thickness_warp); kept as in
        # yunareview for reference
        pairs = []
        for a_idx, b_idx in LIMB_THICKNESS_SEGMENTS[part]:
            seg = _mask_width_perpendicular(
                mask_arr, pose_px[a_idx], pose_px[b_idx], return_points=True, symmetric_fallback=True)
            if seg is None:
                continue
            center = (np.asarray(pose_px[a_idx], dtype=float) + pose_px[b_idx]) / 2
            for pt in seg:
                pairs.append((pt, center + (pt - center) * scale))
        return pairs

    if part in WIDTH_ROW_FRACTIONS:
        top_pair, bottom_pair, frac = WIDTH_ROW_FRACTIONS[part]
        y = mid_y(top_pair) + (mid_y(bottom_pair) - mid_y(top_pair)) * frac
        x_center = mid_x(top_pair) + (mid_x(bottom_pair) - mid_x(top_pair)) * frac
        bounds = _mask_row_bounds_px(mask_arr, y, x_center)
    elif part == "waist_silhouette":
        shoulder_x, hip_x = mid_x((L_SHOULDER, R_SHOULDER)), mid_x((L_HIP, R_HIP))
        shoulder_y, hip_y = mid_y((L_SHOULDER, R_SHOULDER)), mid_y((L_HIP, R_HIP))
        bounds, y = None, None
        for frac in np.linspace(0.35, 0.85, 15):
            row_y = shoulder_y + (hip_y - shoulder_y) * frac
            x_center = shoulder_x + (hip_x - shoulder_x) * frac
            b = _mask_row_bounds_px(mask_arr, row_y, x_center)
            if b and (bounds is None or (b[1] - b[0]) < (bounds[1] - bounds[0])):
                bounds, y = b, row_y
    else:
        return []

    if bounds is None:
        return []
    left_x, right_x = bounds
    center_x = (left_x + right_x) / 2
    return [
        (np.array([left_x, y]), np.array([center_x + (left_x - center_x) * scale, y])),
        (np.array([right_x, y]), np.array([center_x + (right_x - center_x) * scale, y])),
    ]


def _head_hair_control_points(pose_px, mask_arr, pct):
    """머리(헤어 포함) 크기를 pct%만큼 늘리거나 줄인다 — head_silhouette_size()가 재는 것과
    같은 세 기준점(정수리, 눈높이 실루엣 좌우 경계)을 코 위치를 중심으로 방사형으로 밀어낸다.

    예전에 H/B ratio(head_ratio) 보정을 시도했다가 정수리~턱 두 점만 잡고 코를 지나는 선
    하나로 늘려서 눈/코/입까지 같이 딸려 늘어나 실사용 불가 판정으로 뺐다. 이번엔 얼굴
    자체(오벌 등 얼굴 랜드마크)는 이 함수가 아예 건드리지 않고, build_edit_points가 얼굴
    랜드마크 전체를 고정 앵커로 박아서 보호한다 — 여기서 움직이는 점들은 얼굴 오벌 바깥
    (정수리, 헤어 경계)이라 얼굴 앵커와 부딪히지 않는다."""
    if mask_arr is None:
        return []
    nose = np.asarray(pose_px[NOSE], dtype=float)
    l_eye = np.asarray(pose_px[L_EYE], dtype=float)
    r_eye = np.asarray(pose_px[R_EYE], dtype=float)
    nose_x, nose_y = nose
    eye_y = (l_eye[1] + r_eye[1]) / 2

    x_col = int(np.clip(round(nose_x), 0, mask_arr.shape[1] - 1))
    col = np.asarray(mask_arr)[:, x_col].reshape(-1)
    rows = np.where(col > 0.5)[0]
    top_rows = rows[rows < eye_y]
    if len(top_rows) == 0:
        return []
    crown_pt = np.array([nose_x, float(top_rows.min())])

    eye_row = int(np.clip(round(eye_y), 0, mask_arr.shape[0] - 1))
    mask_row = np.asarray(mask_arr)[eye_row].reshape(-1)
    bounds = _mask_row_center_bounds(mask_row, nose_x, 0.5)
    if bounds is None:
        return []
    left_pt = np.array([float(bounds[0]), eye_y])
    right_pt = np.array([float(bounds[1]), eye_y])

    scale = 1 + pct / 100.0
    return [
        (crown_pt, nose + (crown_pt - nose) * scale),
        (left_pt, nose + (left_pt - nose) * scale),
        (right_pt, nose + (right_pt - nose) * scale),
    ]


WHOLE_BODY_MOVED = list(range(33))  # 전신 스케일은 33개 랜드마크 전부를 움직인다


def _whole_body_scale_control_points(pose_px, pct):
    """전신을 골반 중심(hip_mid) 기준으로 pct%만큼 통째로 축소/확대한다.

    "키" 보정(upper+leg_l+leg_r)은 상체(어깨)는 고정하고 그 아래만 늘리는 방식이라, 반복
    적용하면 발이 프레임 밖으로 밀려난다 — 실측으로 확인됨. 키를 늘리기 전에 이 편집으로
    전신을 먼저(예: -5%) 줄여두면(골반 기준 축소라 발이 위로, 머리는 아래로 살짝 당겨져서
    여유 공간이 생김) 그 다음 키 보정이 프레임 밖으로 안 나가고 들어갈 여유가 생긴다.
    두 편집은 한 번에 같이 걸어도 되고, 이 편집 먼저 누르고 재측정된 결과 위에 "키"를
    따로 눌러도 된다(둘 다 누적 워크플로에서 정상 동작).
    """
    hip_mid = (np.asarray(pose_px[L_HIP], dtype=float) + pose_px[R_HIP]) / 2
    scale = 1 + pct / 100.0
    return [(pose_px[i], hip_mid + (np.asarray(pose_px[i], dtype=float) - hip_mid) * scale)
            for i in WHOLE_BODY_MOVED]


def plan_edit(pose_px, mask_arr, part, pct):
    """단일 부위 pct% 보정(양수=늘림/확장, 음수=줄임)에 필요한 이동 제어점
    [(src_pt, dst_pt), ...] 반환. pose_px: (33,2) 픽셀 좌표. 실패시 [].

    H/B ratio(head_ratio)는 일부러 여기 없음 — 정수리~턱을 코를 지나는 선 하나로 늘리면
    얼굴이 눈에 띄게 일그러져서 실사용 불가 판정, 보정 기능은 빼고 측정/표시만 남김.
    head_silhouette(헤어 포함 머리 크기)는 얼굴을 건드리지 않는 방식이라 별도로 지원한다."""
    if part in LENGTH_CHAINS:
        top, bottom, moved = LENGTH_CHAINS[part]
        top_pt = top(pose_px) if callable(top) else pose_px[top]
        bottom_pt = bottom(pose_px) if callable(bottom) else pose_px[bottom]
        offset = (pct / 100.0) * (np.asarray(bottom_pt, dtype=float) - np.asarray(top_pt, dtype=float))
        return [(pose_px[i], np.asarray(pose_px[i], dtype=float) + offset) for i in moved]
    if part in WIDTH_PARTS:
        return _width_control_points(pose_px, mask_arr, part, pct)
    if part == "head_silhouette":
        return _head_hair_control_points(pose_px, mask_arr, pct)
    if part == "whole_body":
        return _whole_body_scale_control_points(pose_px, pct)
    return []


def build_edit_points(pose_px, mask_arr, img_w, img_h, edits, face_px=None):
    """여러 부위 편집(edits: [{"part":.., "pct":..}, ...])을 하나의 MLS 제어점 집합으로 합친다.
    편집 대상이 아닌 랜드마크 전부 + 이미지 네 모서리를 고정 앵커로 넣어 변형을 국소화한다
    (모서리 앵커가 없으면 몸 밖 배경까지 전체적으로 밀릴 수 있음).

    face_px(478,3 픽셀 좌표, pose_px와 같은 원본 이미지 좌표계)를 주면 얼굴 오벌(윤곽) 36점도
    고정 앵커로 추가한다 — "head_silhouette"(헤어 포함 머리 크기) 편집이 정수리/헤어 경계만
    움직이고 얼굴 자체는 이 앵커들 때문에 안 딸려 움직이게 하기 위함. 478점 전부가 아니라
    오벌 경계만 쓰는 이유는 속도 — 경계가 막혀 있으면 MLS 특성상 안쪽(눈/코/입)도 자연히
    안 움직이므로 내부 점까지 앵커로 박을 필요가 없다(478점 다 쓰면 제어점이 너무 많아져서
    보정 하나에 1분 넘게 걸리는 게 실측으로 확인됨). 다른 편집에도 부작용 없이 적용되므로
    (얼굴 근처를 안 건드리면 무관) 항상 넣어도 안전하다 — 단 "whole_body"(전신 스케일)는
    예외: 그 편집은 코/눈 같은 얼굴 근처 포즈 랜드마크까지 같이 움직이므로, 얼굴을 고정
    앵커로 박으면 서로 충돌해 그 부위가 일그러진다. edits에 "whole_body"가 있으면 얼굴
    앵커를 아예 안 넣는다.

    LENGTH_CHAINS/whole_body 편집은 순서대로 적용하며 매 편집이 그 앞의 편집으로 이미 옮겨진
    위치를 기준으로 계산되게 한다(누적) — 예: "키" 보정처럼 상체+양다리를 같은 %로 동시에 걸
    때 "leg_l"은 "upper"가 이미 내려놓은 골반 위치를 기준으로 계산돼야 발목이 두 보정 분량을
    합쳐서 이동한다. 이전엔 모든 편집을 원본 pose_px 기준으로 독립 계산해서 같은 랜드마크를
    나중 편집이 그냥 덮어썼다(외부 코드 리뷰로 지적됨, 재현 확인함) — "키" 보정이 실제로는
    광고한 % 만큼 전신을 늘리지 못하는 버그였다.

    폭 부위(WIDTH_PARTS)와 head_silhouette는 반대로 항상 원본 pose_px 기준으로 계산한다 —
    이 편집들은 mask_arr(실제로 워프되지 않은 원본 마스크)를 스캔해서 경계를 찾는데, 앞선
    길이 편집이 누적한 current_px 위치로 스캔하면 마스크에는 없는(아직 실제로 안 옮겨진)
    자리를 읽게 돼서 엉뚱한 경계를 잡는다(외부 코드 리뷰로 지적됨, 확인함). 한 번에 길이+폭
    편집을 같이 걸어도 이 때문에 안전하지만, 정확도가 중요하면 길이/전신 편집을 먼저 적용해
    재측정한 뒤 폭 편집을 별도 요청으로 거는 편이 더 정확하다(리뷰 권고 그대로).
    반환: (src_pts, dst_pts, warnings)."""
    current_px = np.array(pose_px, dtype=float).copy()  # 편집이 누적되며 갱신되는 "현재" 위치
    moved_idx = set()
    points = {}
    warnings = []
    out_of_frame = False

    for edit in edits:
        part, pct = edit.get("part"), edit.get("pct", 0)
        if not pct:
            continue
        is_length_like = part in LENGTH_CHAINS or part == "whole_body"
        # 폭/머리 편집(마스크 스캔 기반)은 반드시 원본 pose_px를 기준으로 계산해야 한다.
        # mask_arr은 실제로 워프되지 않은 원본 그대로라, 앞선 길이 편집이 누적해둔 current_px
        # 위치로 마스크를 스캔하면 몸통이 실제로 있지도 않은 자리(옮겨졌다고 "가정만" 한 위치)를
        # 읽어서 엉뚱한 경계를 잡는다(코드 리뷰 지적, 확인함) — 길이 편집끼리만 누적이
        # 유효하고, 폭 편집은 항상 원본 좌표 기준.
        reference_px = current_px if is_length_like else pose_px
        pairs = plan_edit(reference_px, mask_arr, part, pct)
        if not pairs:
            warnings.append(f"{part}: 보정 불가(측정 실패 — 마스크/랜드마크 인식 안 됨)")
            continue
        if part in LENGTH_CHAINS or part == "whole_body":
            moved = LENGTH_CHAINS[part][2] if part in LENGTH_CHAINS else WHOLE_BODY_MOVED
            moved_idx.update(moved)
            for i, (_src, dst) in zip(moved, pairs):
                current_px[i] = dst  # 다음 편집이 이 이동을 반영한 위치를 기준으로 계산
        else:
            # 폭 부위: 마스크 경계점이라 랜드마크 인덱스가 없음 — 바로 최종 제어점으로 기록
            for src, dst in pairs:
                points[tuple(np.round(src, 3))] = (src, dst)
                if not (0 <= dst[0] < img_w) or not (0 <= dst[1] < img_h):
                    out_of_frame = True

    # LENGTH_CHAINS로 움직인 랜드마크: 원본 위치(src) -> 모든 편집이 누적된 최종 위치(dst)
    for i in moved_idx:
        src, dst = pose_px[i], current_px[i]
        points[tuple(np.round(src, 3))] = (src, dst)
        if not (0 <= dst[0] < img_w) or not (0 <= dst[1] < img_h):
            out_of_frame = True

    if out_of_frame:
        # 이 판정은 관절 랜드마크(발목/뒤꿈치/발끝) 기준이라, 신발 밑창처럼 랜드마크보다
        # 더 바깥까지 뻗은 실제 실루엣은 랜드마크가 살짝만 넘어도 훨씬 크게 잘려나갈 수 있다
        # (실측 확인 — 발끝 랜드마크는 0.1px만 넘었는데 신발 밑창은 육안으로 뚜렷이 잘림).
        warnings.append("보정 결과 일부가 이미지 프레임 밖으로 나갈 수 있음 — "
                         "\"전신 비율(축소 여유용)\"으로 먼저 살짝 줄여서 여유를 만든 뒤 다시 시도하세요")

    for i in range(33):
        if i in moved_idx:
            continue
        points.setdefault(tuple(np.round(pose_px[i], 3)), (pose_px[i], pose_px[i]))
    has_whole_body = any(e.get("part") == "whole_body" for e in edits)
    if face_px is not None and not has_whole_body:
        # 478점 전부를 앵커로 쓰면 mls_affine_warp의 제어점 루프가 그만큼 느려진다(실측:
        # 머리 보정 하나에 1분 넘게 걸림). 오벌(윤곽) 36점만 고정해도 경계가 막혀 있으면
        # MLS는 안쪽(눈/코/입)도 자연히 안 딸려 움직인다 — 굳이 내부 점까지 다 앵커로 박을
        # 필요 없음. whole_body는 얼굴 근처 포즈 랜드마크도 같이 움직이므로 앵커를 아예 뺀다.
        for i in FACE_GROUPS["oval"]:
            pt = np.asarray(face_px[i][:2], dtype=float)
            points.setdefault(tuple(np.round(pt, 3)), (pt, pt))
    for cx, cy in ((0, 0), (img_w - 1, 0), (0, img_h - 1), (img_w - 1, img_h - 1)):
        points[(cx, cy)] = (np.array([cx, cy], dtype=float), np.array([cx, cy], dtype=float))

    src_pts = np.array([sd[0] for sd in points.values()], dtype=float)
    dst_pts = np.array([sd[1] for sd in points.values()], dtype=float)
    return src_pts, dst_pts, warnings


MODEL_URLS = {
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def _ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)


def _ensure_width_multiple_of_4(pil_img):
    """mediapipe 1.0.1의 segmentation mask 생성이 이미지 폭이 4의 배수가 아니면 파이썬에서
    못 잡는 네이티브 fatal error(Check failed: 1 == ChannelSize())로 프로세스 자체가 죽는다
    (실제로 재현됨). 폭을 살짝(최대 3px) 늘려 우회 — 비율 기반 측정이라 이 정도 왜곡은 무시 가능.
    """
    w, h = pil_img.size
    rem = w % 4
    if rem == 0:
        return pil_img
    pad = 4 - rem
    # 전체 폭을 리사이즈하면 모든 픽셀이 보간된다. 오른쪽 끝 열을 복제해 패딩하면 원본
    # 픽셀은 그대로 유지되어 실제 국소 보정 차이만 남는다.
    result = PILImage.new(pil_img.mode, (w + pad, h))
    result.paste(pil_img, (0, 0))
    edge = pil_img.crop((w - 1, 0, w, h)).resize((pad, h))
    result.paste(edge, (w, 0))
    return result


def _head_crop_box(pose_landmarks_norm, img_w, img_h, margin=3.0):
    """정규화(0~1) 포즈 랜드마크에서 머리 주변 정사각 크롭 박스(px)를 추정. 부족하면 None.

    전신샷은 얼굴이 이미지 대비 작아 얼굴 검출기가 리사이즈 후 놓친다 — 머리 부분만
    미리 잘라내 얼굴 검출기에 넘기면 해결된다.
    """
    pts = []
    for idx in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR):
        lm = pose_landmarks_norm[idx]
        if getattr(lm, "visibility", 1.0) >= 0.3:
            pts.append((lm.x * img_w, lm.y * img_h))
    if len(pts) < 2:
        return None

    xs, ys = zip(*pts)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    head_size = max(max(xs) - min(xs), max(ys) - min(ys), img_w * 0.02)
    half = head_size * margin / 2

    left = max(0, int(cx - half))
    top = max(0, int(cy - half))
    right = min(img_w, int(cx + half))
    bottom = min(img_h, int(cy + half))
    if right - left < 10 or bottom - top < 10:
        return None
    return (left, top, right, bottom)


def _detect_face(face_landmarker, mp_image):
    result = face_landmarker.detect(mp_image)
    return result.face_landmarks[0] if result.face_landmarks else None


# MediaPipe landmarkers are not thread-safe and crash natively when recreated repeatedly:
# create once, and serialize every use with this lock (the HTTP server is threaded).
_pose_landmarker = None
_face_landmarker = None
LOCK = threading.Lock()


def _get_landmarkers():
    global _pose_landmarker, _face_landmarker
    if _pose_landmarker is None:
        _ensure_models()
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_segmentation_masks=True,
        ))
        _face_landmarker = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.3,
        ))
    return _pose_landmarker, _face_landmarker


def detect(rgb_arr):
    """RGB array -> (pose_px (33,2) pixel coords, mask (h,w) float, face_px (478,3) or None).

    Caller holds LOCK. The face is optional: without it head edits just lose their face anchors.
    """
    import mediapipe as mp

    pose_landmarker, face_landmarker = _get_landmarkers()
    pil_img = _ensure_width_multiple_of_4(PILImage.fromarray(rgb_arr))
    full_arr = np.array(pil_img)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=full_arr)
    result = pose_landmarker.detect(image)
    if not result.pose_landmarks or not result.segmentation_masks:
        raise DetectionError("사람(자세/실루엣)을 인식하지 못했습니다.")
    lm2d = result.pose_landmarks[0]
    pose_px = np.array([[p.x * pil_img.width, p.y * pil_img.height] for p in lm2d])
    mask = np.array(result.segmentation_masks[0].numpy_view(), dtype=np.float32, copy=True)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    face_lm, size, offset = _detect_face(face_landmarker, image), (pil_img.width, pil_img.height), (0.0, 0.0)
    if face_lm is None:
        box = _head_crop_box(lm2d, pil_img.width, pil_img.height)
        if box is not None:
            crop = np.array(pil_img.crop(box))
            face_lm = _detect_face(face_landmarker, mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            size, offset = (crop.shape[1], crop.shape[0]), (float(box[0]), float(box[1]))
    face_px = None
    if face_lm is not None:
        face_px = _face_landmarks_to_px(np.array([[p.x, p.y, p.z] for p in face_lm]), *size, *offset)
    return pose_px, mask, face_px


def detect_face(rgb_arr):
    """yunaviewer: face landmarks (478,3) in pixels, or None. Face only, so close-ups where the pose model
    sees no body still work. Caller holds LOCK."""
    import mediapipe as mp

    _, face_landmarker = _get_landmarkers()
    pil_img = _ensure_width_multiple_of_4(PILImage.fromarray(rgb_arr))
    face_lm = _detect_face(face_landmarker, mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_img)))
    if face_lm is None:
        return None
    return _face_landmarks_to_px(np.array([[p.x, p.y, p.z] for p in face_lm]), pil_img.width, pil_img.height)


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def limb_thickness_warp(rgb_arr, pose_px, mask_arr, edits):
    """yunaviewer: smooth thickness change along limb segments (replaces MLS for *_thickness parts).

    MLS with control points on only the silhouette edge made the outline wavy. Here each segment
    gets a displacement field in bone coordinates: t along the bone (0 = upper joint, 1 = lower),
    d = signed distance from the bone axis. Inside the measured half-width W(t) a pixel moves by
    d * (scale - 1); outside it the shift fades to 0 over another W, so the background is not
    torn. The field tapers off at both joints. Fields of different limbs are summed, so legs that
    touch push into each other and cancel at the seam while their outer edges widen.
    Returns (warped, warnings)."""
    h, w = rgb_arr.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    disp_x = np.zeros((h, w), np.float32)
    disp_y = np.zeros((h, w), np.float32)
    warnings = []
    for edit in edits:
        s = edit["pct"] / 100.0
        applied = False
        for a_idx, b_idx in LIMB_THICKNESS_SEGMENTS[edit["part"]]:
            a, b = np.asarray(pose_px[a_idx], dtype=float), np.asarray(pose_px[b_idx], dtype=float)
            length = float(np.linalg.norm(b - a))
            if length < 4:
                continue
            axis = (b - a) / length
            perp = np.array([-axis[1], axis[0]])
            half = (b - a) * 0.02
            ts, widths = [], []
            for t in LIMB_SAMPLE_FRACTIONS:
                mid = a + (b - a) * t
                seg = _mask_width_perpendicular(mask_arr, mid - half, mid + half, return_points=True,
                                                symmetric_fallback=True,
                                                max_half=int(max(80, np.ceil(length * 0.6))))
                if seg is not None:
                    ts.append(t)
                    widths.append(max(np.linalg.norm(seg[0] - mid), np.linalg.norm(seg[1] - mid)))
            if not widths:
                continue
            rx, ry = gx - a[0], gy - a[1]
            t_px = (rx * axis[0] + ry * axis[1]) / length
            d_px = rx * perp[0] + ry * perp[1]
            width = np.interp(t_px, ts, widths).astype(np.float32)
            dist = np.abs(d_px)
            inside = np.minimum(dist, width)  # pixels beyond the edge move like the edge...
            fade = 1.0 - _smoothstep(width, 2 * width, dist)  # ...then fade out over another W
            taper = _smoothstep(-0.05, 0.12, t_px) * (1.0 - _smoothstep(0.88, 1.05, t_px))
            shift = (np.sign(d_px) * inside * s * fade * taper).astype(np.float32)
            disp_x += shift * perp[0]
            disp_y += shift * perp[1]
            applied = True
        if not applied:
            warnings.append(f"{edit['part']}: 보정 불가(측정 실패 — 마스크/랜드마크 인식 안 됨)")
    # backward map approximated with the forward field at the destination (displacements are small)
    warped = cv2.remap(rgb_arr, gx - disp_x, gy - disp_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped, warnings, float(np.abs(disp_x).max() + np.abs(disp_y).max())


FEET = [L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT, R_FOOT]


def _ground_points(pose_px, src_pts, dst_pts, img_w, img_h):
    """yunaviewer: when an edit lifts the feet, the MLS warp drags the ground below them up too and the bottom
    rows sample outside the photo (smeared stripes). If there is room under the feet, pin the bottom edge so
    the ground stretches a little (returns extra anchors, trim 0); if the feet nearly touch the bottom, the
    stretch would be just as ugly, so leave it and return how many bottom rows to trim off instead."""
    lift = 0.0
    for i in FEET:
        hit = np.all(np.isclose(src_pts, pose_px[i]), axis=1)
        if hit.any():
            lift = max(lift, float(src_pts[hit][0][1] - dst_pts[hit][0][1]))
    if lift < 1:
        return np.empty((0, 2)), 0
    # soles: toe/heel landmarks plus a little (the mask is no use here: it runs on into the shadows)
    sole = max(float(pose_px[i][1]) for i in FEET) + img_h * 0.015
    space = img_h - 1 - sole  # ground between the soles and the bottom edge
    if space >= lift:  # the ground stretches by at most 2x
        xs = np.linspace(0, img_w - 1, 9)[1:-1]  # the corners are anchored already
        return np.stack([xs, np.full_like(xs, img_h - 1)], 1), 0
    return np.empty((0, 2)), int(np.ceil(lift * 1.15)) + 4


def edit_image_array(rgb_arr, edits):
    """Apply validated body edits [{part, pct}] to an RGB array. Returns (warped_rgb, warnings, trim_bottom):
    trim_bottom rows at the bottom hold no real content (feet lifted with no ground left to fill) and should be cut.
    Limb thickness uses limb_thickness_warp(); every other part uses yunareview's MLS warp.
    Both run on one detection of the original (thickness changes do not move the joints)."""
    clean = validate_edits(edits)
    rgb_arr = np.ascontiguousarray(np.asarray(rgb_arr, dtype=np.uint8)[:, :, :3])
    with LOCK:
        pose_px, mask, face_px = detect(rgb_arr)
    h, w = rgb_arr.shape[:2]
    mask = mask[:h, :w]
    thickness = [e for e in clean if e["part"] in LIMB_THICKNESS_SEGMENTS]
    others = [e for e in clean if e["part"] not in LIMB_THICKNESS_SEGMENTS]
    warnings, moved, trim = [], 0.0, 0
    result = rgb_arr
    if thickness:
        result, warn, shift = limb_thickness_warp(result, pose_px, mask, thickness)
        warnings += warn
        moved = max(moved, shift)
    if others:
        padded = np.array(_ensure_width_multiple_of_4(PILImage.fromarray(result)))
        src_pts, dst_pts, warn = build_edit_points(pose_px, mask, padded.shape[1], h, others, face_px=face_px)
        warnings += warn
        if len(src_pts) and float(np.linalg.norm(dst_pts - src_pts, axis=1).max()) >= 0.25:
            ground, trim = _ground_points(pose_px, src_pts, dst_pts, w, h)
            src_pts, dst_pts = np.vstack([src_pts, ground]), np.vstack([dst_pts, ground])
            result = mls_affine_warp(padded, src_pts, dst_pts)[:h, :w]
            moved = max(moved, 1.0)
    if moved < 0.25:
        raise DetectionError("적용할 수 있는 체형 보정이 없습니다. " + " / ".join(warnings))
    return result, warnings, min(trim, h // 4)
