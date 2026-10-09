#!/usr/bin/env python3
""".venv/bin/python test_body_edit.py  (no MediaPipe model needed)"""

import numpy as np

import body_edit
import server


def main() -> None:
    # validate_edits: known parts only, non-zero, within +-30%
    assert body_edit.validate_edits([{"part": "thigh_l_thickness", "pct": 6}]) == [{"part": "thigh_l_thickness", "pct": 6.0}]
    for bad in ([], [{"part": "nose", "pct": 5}], [{"part": "arm_l", "pct": 0}], [{"part": "arm_l", "pct": 31}]):
        try:
            body_edit.validate_edits(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass

    # merge_body: the model returns totals; untouched parts keep their value, 0 removes, clamped to 30
    current = [{"part": "thigh_l_thickness", "pct": 6}, {"part": "waist_silhouette", "pct": -10}]
    merged = server.merge_body(current, [{"part": "thigh_l_thickness", "pct": 16}, {"part": "arm_l", "pct": 50}])
    assert merged == [{"part": "thigh_l_thickness", "pct": 16.0}, {"part": "waist_silhouette", "pct": -10.0},
                      {"part": "arm_l", "pct": 30.0}], merged
    assert server.merge_body(current, [{"part": "thigh_l_thickness", "pct": 0}]) == [{"part": "waist_silhouette", "pct": -10.0}]

    # MLS warp: fixed points keep the image, a moved point pulls content with it
    img = np.zeros((120, 160, 3), np.uint8)
    img[50:70, 70:90] = 255  # white square centered at (80, 60)
    corners = [(0, 0), (159, 0), (0, 119), (159, 119)]
    same = body_edit.mls_affine_warp(img, corners + [(80, 60)], corners + [(80, 60)])
    assert np.abs(same.astype(int) - img).max() <= 1
    moved = body_edit.mls_affine_warp(img, corners + [(80, 60)], corners + [(100, 60)])
    assert moved[60, 100].min() > 200 and moved[60, 70].max() < 50, (moved[60, 100], moved[60, 70])
    # limb thickness field: a vertical "thigh" (hip at top, knee at bottom) widens evenly along its length
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, 80:120] = 255  # 40 px wide limb
    mask = (img[:, :, 0] > 0).astype(np.float32)
    pose = np.zeros((33, 2))
    pose[body_edit.L_HIP] = (100, 20)
    pose[body_edit.L_KNEE] = (100, 180)
    out, warn, _ = body_edit.limb_thickness_warp(img, pose, mask, [{"part": "thigh_l_thickness", "pct": 20}])
    widths = [int((out[y, :, 0] > 127).sum()) for y in (60, 100, 140)]
    assert not warn and all(46 <= w <= 50 for w in widths), widths  # 40 * 1.2 = 48 at every height
    assert int((out[5, :, 0] > 127).sum()) == 40  # tapers back to the original beyond the joint

    # resize_legs: band hip+15%..ankle is resampled, rows above and below stay as they were
    img = np.arange(200, dtype=np.uint8)[:, None, None].repeat(40, 1).repeat(3, 2)  # row y holds value y
    pose = np.zeros((33, 2))
    pose[body_edit.L_HIP] = pose[body_edit.R_HIP] = (20, 40)
    pose[body_edit.L_ANKLE] = pose[body_edit.R_ANKLE] = (20, 160)  # band rows 58..160 (102 rows)
    pose[body_edit.L_KNEE] = pose[body_edit.R_KNEE] = (20, 100)
    real_detect, body_edit.detect = body_edit.detect, lambda rgb: (pose, None, None)
    try:
        longer = body_edit.resize_legs(img, 20)
        assert longer.shape[0] == 200 + round(102 * 0.2) and (longer[:58] == img[:58]).all()
        assert (longer[-40:] == img[-40:]).all()
        shorter = body_edit.resize_legs(np.dstack([img, img[:, :, :1]]), -20)  # RGBA input keeps its channels
        assert shorter.shape == (200 - round(102 * 0.2), 40, 4), shorter.shape
        def rejected():
            try:
                body_edit.resize_legs(img, 10)
            except body_edit.DetectionError:
                return True
            return False
        pose[body_edit.L_ANKLE] = pose[body_edit.R_ANKLE] = (20, 50)  # ankles right under the hips
        assert rejected()
        pose[body_edit.L_ANKLE] = pose[body_edit.R_ANKLE] = (20, 160)
        pose[body_edit.L_KNEE] = pose[body_edit.R_KNEE] = (20, 30)  # knees above the hips: sitting/folded
        assert rejected()
        pose[body_edit.L_KNEE] = pose[body_edit.R_KNEE] = (20, 100)
        pose[body_edit.L_ANKLE] = (20, 90)  # one foot raised: not standing
        assert rejected()
        pose[body_edit.L_ANKLE] = (20, 160)
        pose[body_edit.R_ANKLE] = (220, 160)  # legs far from vertical (sitting, legs forward)
        assert rejected()
    finally:
        body_edit.detect = real_detect
    print("ok")


if __name__ == "__main__":
    main()
