#!/usr/bin/env python3
""".venv/bin/python test_library.py"""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import library


def full_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_image(path: Path, seed: int, size=(120, 90)) -> None:
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        (root / "v1").mkdir()
        (root / "other").mkdir()
        for i, name in enumerate(["a.png", "b.png", "c.png"]):
            noise_image(root / "v1" / name, i)

        # favorites follow the content, not the path
        a = root / "v1" / "a.png"
        library.set_favorites(root, [a], True)
        assert library.quick_hash(a) in library.load_favorites(root)
        moved = root / "other" / "renamed.png"
        a.rename(moved)
        assert library.quick_hash(moved) in library.load_favorites(root)
        library.set_favorites(root, [moved], False)
        assert not library.load_favorites(root)
        moved.rename(a)

        # ratings and notes: set, keep the other field, clear, follow a renamed file
        assert library.set_marks(root, [a], rating=4) == 1
        library.set_marks(root, [a], note="  best pose  ")
        assert library.load_marks(root)[library.quick_hash(a)] == {"rating": 4, "note": "best pose"}
        library.set_marks(root, [a], rating=0)
        assert library.load_marks(root)[library.quick_hash(a)] == {"note": "best pose"}
        library.set_marks(root, [a], note="")
        assert library.quick_hash(a) not in library.load_marks(root)
        try:
            library.set_marks(root, [a], rating=6)
            raise AssertionError("rating 6 accepted")
        except ValueError:
            pass
        assert len(library.set_marks(root, [a], note="x" * 500) and library.load_marks(root)[library.quick_hash(a)]["note"]) == library.MAX_NOTE
        library.set_marks(root, [a], note="")

        # trash -> undo, and restore from the trash folder
        op = library.trash_files(root, [a, root / "v1" / "b.png"])
        assert not a.exists() and len(list((root / ".yunaviewer" / "trash").iterdir())) == 2
        assert library.undo_op(root, op["id"]) == {"restored": 2, "skipped": 0, "renamed": 0}
        assert a.exists()
        library.trash_files(root, [a])
        trashed = next((root / ".yunaviewer" / "trash").iterdir())
        assert library.restore_from_trash(root, [trashed]) == 1 and a.exists()
        # emptying deletes the trashed files for good and forgets their undo entries; the rest stays
        noise_image(root / "gone.png", 7)
        op = library.trash_files(root, [root / "gone.png"])
        assert library.empty_trash(root) == 1 and not list((root / ".yunaviewer" / "trash").iterdir())
        assert a.exists() and op["id"] not in {o["id"] for o in json.loads((root / ".yunaviewer" / "ops.json").read_text())}
        assert library.empty_trash(root) == 0
        # delete only selected trashed files; anything outside the trash is refused
        for n in ("keep.png", "drop.png"):
            noise_image(root / n, 8)
        library.trash_files(root, [root / "keep.png", root / "drop.png"])
        trash_files = {p.name.split("__", 1)[1]: p for p in (root / ".yunaviewer" / "trash").iterdir()}
        assert library.empty_trash(root, [trash_files["drop.png"]]) == 1
        assert [p.name.split("__", 1)[1] for p in (root / ".yunaviewer" / "trash").iterdir()] == ["keep.png"]
        try:
            library.empty_trash(root, [a])
            raise AssertionError("deleted a file outside the trash")
        except ValueError:
            assert a.exists()
        library.empty_trash(root)

        # move never overwrites; undo puts files back
        noise_image(root / "other" / "a.png", 99)
        op = library.move_files(root, [a], root / "other")
        assert (root / "other" / "a_2.png").exists() and (root / "other" / "a.png").exists()
        library.undo_op(root, op["id"])
        assert a.exists() and not (root / "other" / "a_2.png").exists()
        for bad in (lambda: library.move_files(root, [a], root / ".yunaviewer"),
                    lambda: library.trash_files(root, [root / ".yunaviewer" / "favorites.json"])):
            try:
                bad()
                raise AssertionError("allowed a state path")
            except ValueError:
                pass
        # Validate the whole batch before moving anything, so a later bad item cannot cause a partial operation.
        before = a.read_bytes()
        for operation in (lambda: library.trash_files(root, [a, root / "missing.png"]),
                          lambda: library.move_files(root, [a, root / "missing.png"], root / "other")):
            try:
                operation()
                raise AssertionError("partially valid batch was accepted")
            except ValueError:
                assert a.is_file() and a.read_bytes() == before
        # A filesystem failure after the first move rolls earlier items back as well.
        b = root / "v1" / "b.png"
        real_move, move_calls = library.shutil.move, []

        def fail_second_move(source, target):
            move_calls.append((source, target))
            if len(move_calls) == 2:
                raise OSError("injected move failure")
            return real_move(source, target)
        library.shutil.move = fail_second_move
        try:
            try:
                library.move_files(root, [a, b], root / "other")
                raise AssertionError("injected move failure was ignored")
            except OSError:
                assert a.is_file() and b.is_file()
                assert not (root / "other" / "a_2.png").exists()
        finally:
            library.shutil.move = real_move
        assert library.list_dirs(root) == ["", "other", "v1"]
        # a symlinked top-level folder is a mounted library: its sub folders are listed, inner links are not followed
        outside = root.parent / (root.name + "_mount")
        (outside / "deep").mkdir(parents=True)
        (root / "mounted").symlink_to(outside, target_is_directory=True)
        (outside / "deep" / "loop").symlink_to(outside, target_is_directory=True)
        assert library.list_dirs(root) == ["", "mounted", "mounted/deep", "other", "v1"], library.list_dirs(root)
        (root / "mounted").unlink()
        # undoing into a name that is taken again keeps both files and reports it
        busy = library.trash_files(root, [root / "v1" / "c.png"])
        noise_image(root / "v1" / "c.png", 123)
        assert library.undo_op(root, busy["id"]) == {"restored": 1, "skipped": 0, "renamed": 1}
        assert (root / "v1" / "c_2.png").exists()
        assert library.make_dir(root, root, "new") == root / "new"

        # albums from descriptions
        def tags(top, bottom="not visible", pose="standing", hair="long brown wavy hair with bangs", visible="fullbody",
                 onepiece="none", outfit=None):
            t = {"top": top, "bottom": bottom, "pose": pose, "hair": hair, "visible": visible, "onepiece": onepiece,
                 "outer": "none"}
            if outfit:
                t["filename_outfit"] = outfit
            return t
        entries = [
            ("1", tags("cream sheer long-sleeve blouse", "light blue long flared skirt", outfit="blouseskirt")),
            ("2", tags("cream long-sleeve blouse", "light blue flared skirt", outfit="blouseskirt")),
            ("3", tags("cream sheer long-sleeve blouse", visible="upperbody", hair="short dark brown bob, tied")),
            ("4", tags("none", onepiece="red off-the-shoulder knit mini dress", pose="sitting on bed")),
            ("5", tags("none", onepiece="red knit off-the-shoulder dress", pose="lying down")),
        ]
        outfits = {tuple(g["names"]): g["label"] for g in library.build_albums("outfit", entries)}
        assert ("1", "2", "3") in {tuple(sorted(k)) for k in outfits}, outfits  # cropped shot joins its outfit
        assert ("4", "5") in {tuple(sorted(k)) for k in outfits}
        assert "블라우스" in library.build_albums("outfit", entries)[0]["label"]
        assert library.garment_label("black lace bra") == "검정 레이스 브라"
        assert library.garment_label("short red floral kimono dress white cherry blossom pattern") == \
            "빨강 흰색 벚꽃무늬 기모노 원피스"
        assert not library.same_outfit(entries[0][1], entries[3][1])  # blouse+skirt is not a dress

        # same garment worded differently is one outfit; a different colour or garment type is not
        def one(top, bottom="none", onepiece="none", **extra):
            return {"top": top, "bottom": bottom, "onepiece": onepiece, "outer": "none", **extra}
        jersey_a = one("white oversized short-sleeve sports jersey with red and blue graphics")
        jersey_b = one("white branded short-sleeve sports jersey top", "not visible")
        assert library.same_outfit(jersey_a, jersey_b)
        assert library.same_outfit(one("cream knit sweater"), one("ivory knitted long-sleeve sweater top"))
        assert not library.same_outfit(one("cream knit sweater"), one("black knit sweater"))
        assert not library.same_outfit(one("none", onepiece="red knit mini dress"), one("red knit sweater", "grey skirt"))
        assert not library.same_outfit(one("red knit dress top"), one("red knit sweater vest"))
        # photos with the same myshare label match regardless of wording; different labels fall back to the text
        assert library.same_outfit({**jersey_a, "filename_outfit": "x"}, one("pink hoodie", filename_outfit="x"))
        assert not library.same_outfit({**jersey_a, "filename_outfit": "x"}, one("pink hoodie", filename_outfit="y"))
        poses = {g["label"]: g["names"] for g in library.build_albums("pose", entries)}
        assert poses["서 있음"] == ["1", "2", "3"] and poses["앉아 있음"] == ["4"] and poses["누워 있음"] == ["5"]
        hair = {g["label"]: sorted(g["names"]) for g in library.build_albums("hair", entries)}
        assert any("짧은 머리" in label and "묶음" in label for label in hair), hair
        assert any("긴 머리 · 갈색" in label for label in hair), hair

        # duplicates: byte-identical copy and a slightly re-encoded look-alike, but not an unrelated image
        folder = root / "dups"
        folder.mkdir()
        noise_image(folder / "orig.png", 5)
        (folder / "copy.png").write_bytes((folder / "orig.png").read_bytes())
        base = np.asarray(Image.open(folder / "orig.png")).astype(int)
        Image.fromarray(np.clip(base + 2, 0, 255).astype(np.uint8)).save(folder / "similar.png")
        noise_image(folder / "different.png", 77)
        groups = library.find_duplicates(root, sorted(folder.iterdir()), lambda p: p, full_hash)
        assert len(groups) == 1 and groups[0]["count"] == 3, groups
        assert set(groups[0]["names"]) == {"orig.png", "copy.png", "similar.png"}
        exact_only = library.find_duplicates(root, [folder / "orig.png", folder / "copy.png"], lambda p: p, full_hash)
        assert exact_only[0]["kind"] == "exact"
        many_exact = []
        for i in range(13):
            copy = folder / f"many_{i}.png"
            copy.write_bytes((folder / "orig.png").read_bytes())
            many_exact.append(copy)
        assert library.find_duplicates(root, many_exact, lambda p: p, full_hash)[0]["kind"] == "exact"

        # myshare names: <folder>_<outfit>_<shot>_<ratio>_<res>_<seq>, continuing existing numbers
        shop = root / "shop"
        shop.mkdir()
        for i in range(3):
            noise_image(shop / f"pending_{i}.png", 10 + i, size=(90, 160))
        noise_image(shop / "shop_redknitdress_fullbody_9x16_1k_01.png", 20, size=(90, 160))
        items = [{"name": f"pending_{i}.png", "outfit": "Red Knit Dress", "shot": "fullbody"} for i in range(2)]
        items.append({"name": "pending_2.png", "outfit": "x", "shot": "fullbody"})  # token too short: skipped
        plan = library.plan_names(shop, items)
        assert [p["new_name"] for p in plan] == ["shop_redknitdress_fullbody_9x16_1k_02.png",
                                                 "shop_redknitdress_fullbody_9x16_1k_03.png"], plan
        assert library.known_outfits(shop) == ["redknitdress"]
        # anything not in myshare form needs a name, pending_ or not; edited copies of named files do not
        assert [library.needs_name(n) for n in ("pending_1.png", "Seedream5.0_lite_00001 (41).png", "IMG_0001.JPG",
                "shop_redknitdress_fullbody_9x16_1k_01.png", "shop_redknitdress_fullbody_9x16_1k_01_edit.png",
                "shop_redknitdress_fullbody_9x16_1k_01_edit3.jpg")] == [True, True, True, False, False, False]
        op = library.apply_rename(root, shop, plan)
        assert not (shop / "pending_0.png").exists() and (shop / plan[0]["new_name"]).exists()
        library.undo_op(root, op["id"])
        assert (shop / "pending_0.png").exists() and not (shop / plan[0]["new_name"]).exists()
        (shop / plan[1]["new_name"]).write_bytes(b"x")  # target exists: nothing is renamed
        try:
            library.apply_rename(root, shop, plan)
            raise AssertionError("renamed over an existing file")
        except ValueError:
            assert (shop / "pending_0.png").exists() and (shop / "pending_1.png").exists()
    # generation info: ComfyUI graph (prompt, negative via the sampler, model, seed), WebUI text, AIGC label
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "juggernaut.safetensors"}},
             "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a woman in a red dress", "clip": ["1", 1]}},
             "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["1", 1]}},
             "4": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 30, "cfg": 4.0, "positive": ["2", 0],
                                                         "negative": ["3", 0], "model": ["1", 0]}},
             "5": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
             "6": {"class_type": "GrokImageEditNodeV2", "_meta": {"title": "Grok Image Edit"},
                   "inputs": {"prompt": "change the background", "model": "grok-2"}}}
    info = library.comfy_info(graph)
    assert info["prompts"] == ["a woman in a red dress", "change the background"], info
    assert info["negative"] == ["blurry, low quality"] and info["seed"] == 42 and info["input_images"] == 1
    assert info["models"] == ["juggernaut.safetensors", "grok-2"] and info["generators"] == ["KSampler", "Grok Image Edit"]
    assert info["settings"]["steps"] == 30
    class Fake:
        def __init__(self, meta): self.info = meta
    webui = library.generation_info(Fake({"parameters": "cat on a sofa\nNegative prompt: dog\nSteps: 20, Seed: 7"}))
    assert webui["prompts"] == ["cat on a sofa"] and webui["negative"] == ["dog"] and "Seed: 7" in webui["settings"]["parameters"]
    aigc = library.generation_info(Fake({"XML:com.adobe.xmp": '<x>TC260:AIGC>{"Label":"1","ContentProducer":"ABC"}</x>'}))
    assert aigc == {"tool": "AI 생성 표시 (TC260 AIGC)", "producer": "ABC"}
    assert library.generation_info(Fake({})) == {} and library.generation_info(Fake({"prompt": "not json"})) == {}
    print("ok")


if __name__ == "__main__":
    main()
