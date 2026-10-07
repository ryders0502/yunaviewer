#!/usr/bin/env python3
""".venv/bin/python test_effects.py  (synthetic images, no model needed)"""

import cv2
import numpy as np
from PIL import Image

import body_edit
import effects


def main() -> None:
    # orient: flip then rotate clockwise; 90/270 swap the size
    img = Image.new("RGB", (4, 2))
    img.putpixel((0, 0), (255, 0, 0))  # top-left red
    assert effects.orient(img, 90).size == (2, 4)
    assert effects.orient(img, 90).getpixel((1, 0)) == (255, 0, 0)  # top-left goes to top-right
    assert effects.orient(img, 180).getpixel((3, 1)) == (255, 0, 0)
    assert effects.orient(img, 0, True).getpixel((3, 0)) == (255, 0, 0)  # mirrored
    assert effects.orient(img, 0, False) is img

    # background: a bright square "person" on a noisy backdrop
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    base[60:140, 110:190] = 200
    mask = np.zeros((200, 300), np.float32)
    mask[60:140, 110:190] = 1.0
    photo = Image.fromarray(base)
    blurred = np.asarray(effects.apply_background(photo, mask, "blur", 0.5))
    assert blurred[:40].std() < base[:40].std() / 2  # backdrop smoothed
    assert np.abs(blurred[80:120, 130:170].astype(int) - 200).max() <= 1  # person untouched
    removed = effects.apply_background(photo, mask, "remove")
    alpha = np.asarray(removed.getchannel("A"))
    assert removed.mode == "RGBA" and alpha[:30].max() == 0 and alpha[90:110, 140:160].min() == 255
    assert effects.apply_background(photo, mask, "none") is photo

    # face: points on a circle for the oval indices; brightness changes the inside only
    face = np.zeros((478, 3))
    for k, idx in enumerate(body_edit.FACE_OVAL):
        angle = 2 * np.pi * k / len(body_edit.FACE_OVAL)
        face[idx] = (150 + 40 * np.cos(angle), 100 + 40 * np.sin(angle), 0)
    skin = Image.fromarray(np.clip(rng.normal(120, 10, (200, 300, 3)), 0, 255).astype(np.uint8))
    lighter = np.asarray(effects.adjust_face(skin, face, brightness=1.4)).astype(float)
    original = np.asarray(skin).astype(float)
    assert lighter[90:110, 140:160].mean() > original[90:110, 140:160].mean() * 1.3
    assert np.abs(lighter[:30] - original[:30]).max() == 0  # outside the face: identical
    warm = np.asarray(effects.adjust_face(skin, face, tone=100)).astype(float)
    assert warm[90:110, 140:160, 0].mean() > original[90:110, 140:160, 0].mean() * 1.1
    assert warm[90:110, 140:160, 2].mean() < original[90:110, 140:160, 2].mean() * 0.9
    smooth = np.asarray(effects.adjust_face(skin, face, smooth=1.0)).astype(float)
    assert smooth[90:110, 140:160].std() < original[90:110, 140:160].std() * 0.8  # skin noise reduced
    assert np.abs(smooth[:30] - original[:30]).max() == 0
    assert effects.adjust_face(skin, face) is skin

    # face shape on a drawn face: oval r=150 around (300, 400), eyes, nose and a lower lip band
    big = np.zeros((478, 3))
    for k, idx in enumerate(body_edit.FACE_OVAL):
        angle = 2 * np.pi * k / len(body_edit.FACE_OVAL)
        big[idx] = (300 + 120 * np.cos(angle), 400 + 150 * np.sin(angle), 0)
    big[152] = (300, 550, 0)  # chin
    big[234], big[454], big[172], big[397] = (180, 400, 0), (420, 400, 0), (200, 480, 0), (400, 480, 0)
    for eye, cx in ((effects.LEFT_EYE, 250), (effects.RIGHT_EYE, 350)):
        for k, idx in enumerate(eye):
            angle = 2 * np.pi * k / len(eye)
            big[idx] = (cx + 20 * np.cos(angle), 360 + 8 * np.sin(angle), 0)
    big[33], big[133], big[362], big[263] = (230, 360, 0), (270, 360, 0), (330, 360, 0), (370, 360, 0)
    big[[1, 2, 98, 327]] = (300, 420, 0), (300, 430, 0), (288, 425, 0), (312, 425, 0)
    big[129], big[358] = (280, 425, 0), (320, 425, 0)
    big[61], big[291], big[13], big[0], big[14], big[17] = (270, 470, 0), (330, 470, 0), (300, 466, 0), (300, 458, 0), (300, 470, 0), (300, 482, 0)
    drawn = np.full((800, 600, 3), 200, np.uint8)
    cv2.circle(drawn, (300, 425), 12, (0, 0, 0), -1)  # nose
    cv2.circle(drawn, (250, 360), 10, (0, 0, 0), -1)  # left eye
    drawn[470:482, 280:321] = 0  # lower lip
    drawn[380:520, 415:418] = 0  # vertical line on the right cheek/jaw
    portrait = Image.fromarray(drawn)
    dark = lambda a, box: int((np.asarray(a)[box[1]:box[3], box[0]:box[2], 0] < 100).sum())
    assert effects.face_too_small(big, 800) == ""
    assert effects.face_too_small(None, 800) and effects.face_too_small(big * [0.3, 0.3, 1], 800)  # missing / small face
    assert effects.face_too_small(big, 4000)  # 300px face in a 4000px tall photo: under 12%
    nose_box, eye_box, lip_box = (270, 395, 331, 456), (225, 335, 276, 386), (275, 462, 326, 500)
    assert dark(effects.reshape_face(portrait, big, {"nose_size": -20}), nose_box) < dark(portrait, nose_box) * 0.8
    assert dark(effects.reshape_face(portrait, big, {"eye_size": 20}), eye_box) > dark(portrait, eye_box) * 1.2
    assert dark(effects.reshape_face(portrait, big, {"lower_lip": -20}), lip_box) < dark(portrait, lip_box) * 0.9
    slim = np.asarray(effects.reshape_face(portrait, big, {"jaw_width": -20}))
    assert (slim[480, :, 0] < 100).nonzero()[0].mean() < 415  # the line moved toward the face center at the jaw
    assert (slim[300, :, 0] < 100).sum() == 0 and np.array_equal(slim[:150], drawn[:150])  # above: untouched
    assert effects.reshape_face(portrait, big, {}) is portrait and effects.reshape_face(portrait, big, {"x": 5}) is portrait
    # best shot: sharper wins, closed eyes and blown exposure lose, measures are relative within the group
    sharp_img = Image.fromarray(rng.integers(0, 255, (120, 120, 3), dtype=np.uint8))
    soft_img = Image.fromarray(cv2.GaussianBlur(np.asarray(sharp_img), (0, 0), 3))
    m_sharp, m_soft = effects.shot_measures(sharp_img, None), effects.shot_measures(soft_img, None)
    assert m_sharp["sharpness"] > m_soft["sharpness"] * 5 and m_sharp["eyes"] is None
    ranked = effects.rank_shots([{**m_sharp, "pixels": 100}, {**m_soft, "pixels": 100}])
    assert ranked[0]["score"] > ranked[1]["score"] and "가장 선명" in ranked[0]["reasons"] and "흐림" in ranked[1]["reasons"]
    base = {"sharpness": 100.0, "clipped": 0.0, "pixels": 100}
    ranked = effects.rank_shots([{**base, "eyes": 0.3}, {**base, "eyes": 0.05}, {**base, "eyes": 0.3, "clipped": 0.2}])
    assert ranked[0]["score"] > ranked[1]["score"] and "눈 감김" in ranked[1]["reasons"]
    assert ranked[0]["score"] > ranked[2]["score"] and "노출 과다/부족" in ranked[2]["reasons"]
    assert abs(effects.eye_openness(big) - 0.4) < 0.01  # drawn eyes: 16 tall / 40 wide
    # spot removal: a dark dot on flat skin is filled in, pixels far from it stay as they were
    flat = np.full((200, 200, 3), (220, 180, 160), np.uint8)
    spotted = flat.copy()
    cv2.circle(spotted, (100, 100), 6, (40, 20, 20), -1)
    fixed = np.asarray(effects.erase_spots(Image.fromarray(spotted), [{"r": 10, "pts": [[100, 100]]}])).astype(int)
    assert np.abs(fixed[90:111, 90:111] - flat[90:111, 90:111]).max() < 12  # dot gone
    assert np.array_equal(fixed[:40], spotted[:40])  # far away: untouched
    line = effects.erase_mask((200, 200), [{"r": 3, "pts": [[20, 20], [180, 20]]}])
    assert line[20, 100] == 255 and line[40, 100] == 0 and line[20, 190] == 0
    plain = Image.fromarray(spotted)
    assert effects.erase_spots(plain, []) is plain and effects.erase_spots(plain, [{"r": 5, "pts": []}]) is plain
    print("ok")


if __name__ == "__main__":
    main()
