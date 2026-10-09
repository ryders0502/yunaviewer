#!/usr/bin/env python3
"""python3 test_server.py"""

import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import urllib.error
import urllib.request

from PIL import Image

import server


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        server.ROOT = Path(tmp).resolve()
        (server.ROOT / "sub").mkdir()
        for name, size in [("a.jpg", (40, 80)), ("b.png", (90, 30)), ("sub/c.jpg", (10, 10))]:
            Image.new("RGB", size, "red").save(server.ROOT / name)
        (server.ROOT / "note.txt").write_text("x")
        os.utime(server.ROOT / "a.jpg", (2000, 2000))
        os.utime(server.ROOT / "b.png", (1000, 1000))

        for bad in ("../etc/passwd", "/../..", "sub/../../x"):
            try:
                server.safe_path(bad)
                raise AssertionError(f"traversal allowed: {bad}")
            except PermissionError:
                pass

        # A directory symlink deliberately mounted directly below ROOT is a valid virtual folder.
        with tempfile.TemporaryDirectory() as linked_tmp:
            mount = server.ROOT / "myshare"
            mount.symlink_to(linked_tmp, target_is_directory=True)
            linked = Path(linked_tmp)
            source = server.ROOT / "to_myshare.jpg"
            Image.new("RGB", (4, 4), "green").save(source)
            assert server.safe_path("myshare") == mount
            op = server.library.move_files(server.ROOT, [source], server.safe_path("myshare"))
            assert (linked / source.name).is_file()
            server.library.undo_op(server.ROOT, op["id"])
            assert source.is_file()
            source.unlink()
            mount.unlink()

        listing = server.list_dir("")
        sizes = {n: (server.ROOT / n).stat().st_size for n in ("a.jpg", "b.png")}
        assert listing == {"dir": "", "dirs": ["sub"], "files": [  # newest first; no EXIF: taken = file time
            {"name": "a.jpg", "w": 40, "h": 80, "mtime": 2000, "bytes": sizes["a.jpg"], "taken": 2000, "fav": False},
            {"name": "b.png", "w": 90, "h": 30, "mtime": 1000, "bytes": sizes["b.png"], "taken": 1000, "fav": False}],
            "trashed": 0}, listing
        # EXIF capture time wins over the file time
        exif = Image.Exif()
        exif[0x0132] = "2021:05:06 07:08:09"
        Image.new("RGB", (20, 20)).save(server.ROOT / "shot.jpg", exif=exif)
        taken = next(f for f in server.list_dir("")["files"] if f["name"] == "shot.jpg")["taken"]
        assert time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(taken)) == "2021-05-06 07:08:09", taken
        (server.ROOT / "shot.jpg").unlink()
        server.library.set_favorites(server.ROOT, [server.ROOT / "a.jpg"], True)
        server.library.set_marks(server.ROOT, [server.ROOT / "a.jpg"], rating=3, note="keep")
        marked = {f["name"]: f for f in server.list_dir("")["files"]}
        assert marked["a.jpg"]["rating"] == 3 and marked["a.jpg"]["note"] == "keep" and "rating" not in marked["b.png"]
        server.library.set_marks(server.ROOT, [server.ROOT / "a.jpg"], rating=0, note="")
        assert [f["fav"] for f in server.list_dir("")["files"]] == [True, False]
        assert server.list_dir("sub")["dir"] == "sub"
        assert server.thumbnail(server.ROOT / "a.jpg").is_file()
        # Flattening relative paths used to collide: a/b.jpg and a__b.jpg must have distinct cache entries.
        nested = server.ROOT / "sub" / "collision.jpg"
        flat = server.ROOT / "sub__collision.jpg"
        Image.new("RGB", (8, 8), "red").save(nested)
        Image.new("RGB", (8, 8), "blue").save(flat)
        assert server.thumbnail(nested) != server.thumbnail(flat)
        nested.unlink()
        flat.unlink()

        p = Path("v3_plaidtrimcroptop_upperbody_3x4_1k_02.png")
        assert server.search_tags(p, {"top": "x"}) == {"top": "x", "filename_outfit": "plaidtrimcroptop"}
        assert server.search_tags(Path("pending_20261002_224017_108953_81.png"), {"top": "x"}) == {"top": "x"}

        matches = server.map_matches({"matches": ["img-001", "IMG-999", "IMG-001"]},
                                     {"IMG-001": "b.png"}, ["a.jpg"])
        assert matches == ["a.jpg", "b.png"], matches
        assert server.verified_names({"verdicts": [{"id": "img-001", "match": True}, {"id": "IMG-009", "match": True},
                                                   {"id": "IMG-002", "match": "true"}]},
                                     {"IMG-001": "x", "IMG-002": "y"}) == {"x"}

        # describe -> stored tags -> text search -> image verify, with a fake agent
        for i in range(3):
            Image.new("RGB", (5, 5)).save(server.ROOT / f"z{i}.jpg")
            os.utime(server.ROOT / f"z{i}.jpg", (3000 + i, 3000 + i))
        calls = []
        describe_sizes = []

        def fake_agent(job_dir, request):
            calls.append(request.get("mode", "search"))
            if request.get("mode") == "describe":
                describe_sizes.append(len(request["image_ids"]))
                if len(request["image_ids"]) > 2:
                    raise RuntimeError("StructuredOutputException: forced tool failed")
                assert (job_dir / "contact_01.jpg").is_file()
                return {"images": [{"id": i, "top": "red" if n == 0 else "blue"}
                                   for n, i in enumerate(request["image_ids"])]}
            if request.get("mode") == "verify":  # rejects the first verified image
                assert all((job_dir / ref).is_file() for ref in request["references"])
                return {"verdicts": [{"id": i, "match": n > 0} for n, i in enumerate(request["image_ids"])]}
            assert all(t["top"] in ("red", "blue") for t in request["reference_tags"].values())
            return {"matches": list(request["candidate_tags"]), "message": "m", "reference_summary": "s"}
        server.invoke_agent = fake_agent

        # Existing tags are loaded once for the whole folder, not once per image.
        original_stored_tags = server.stored_tags
        stored_calls = []
        server.stored_tags = lambda paths: (stored_calls.append(list(paths)) or {p: {"top": "x"} for p in paths})
        server.describe_images([server.ROOT / "a.jpg", server.ROOT / "b.png"], {})
        assert len(stored_calls) == 1
        server.stored_tags = original_stored_tags

        server.start_tagging(server.ROOT)
        while server.tag_status(server.ROOT)["running"] or not calls:
            time.sleep(0.01)
        assert server.tag_status(server.ROOT)["done"] == 5, server.tag_status(server.ROOT)
        assert len(server.stored_tags(server.folder_images(server.ROOT))) == 5
        assert describe_sizes[0] == 5 and 1 in describe_sizes and 2 in describe_sizes
        snapshot = server.tag_status_snapshot(server.ROOT)
        assert snapshot["done"] == snapshot["total"] == 5 and not snapshot["running"]

        calls.clear()
        result = server.run_search("", "같은 의상", ["a.jpg"])
        assert calls == ["search", "verify"], calls  # nothing re-described
        assert result["matches"] == ["a.jpg", "z1.jpg", "z0.jpg", "b.png"], result

        (server.ROOT / "sub" / "moved").mkdir()  # moved + renamed file keeps its tags
        (server.ROOT / "z0.jpg").rename(server.ROOT / "sub" / "moved" / "renamed.jpg")
        assert len(server.stored_tags([server.ROOT / "sub" / "moved" / "renamed.jpg"])) == 1

        Image.new("RGB", (60, 60), "green").save(server.ROOT / "b.png")  # changed content: needs a new description
        # photos reach the model only on request: without allow_index nothing is analyzed or uploaded
        calls.clear()
        try:
            server.run_search("", "앉은 자세", [])
            raise AssertionError("searched with an undescribed photo and no consent")
        except server.NeedsIndex as error:
            assert error.count == 1, error.count
        assert calls == [], calls
        server.run_search("", "앉은 자세", [], allow_index=True)
        assert calls == ["describe", "search", "verify"], calls
        # a fully described folder needs no consent at all
        calls.clear()
        server.run_search("", "앉은 자세", [])
        assert "describe" not in calls, calls
        # follow-up search: only the given results are candidates, and the model hears the previous request
        seen = {}
        def follow_up(job_dir, request):
            seen.setdefault("prompt", request["prompt"])
            seen.setdefault("count", len(request.get("candidate_tags", {})))
            return fake_agent(job_dir, request)
        server.invoke_agent = follow_up
        result = server.run_search("", "그중 앉은 것만", [], within=["a.jpg", "b.png"], previous="흰 블라우스")
        assert seen["count"] == 2 and "흰 블라우스" in seen["prompt"], seen
        assert set(result["matches"]) <= {"a.jpg", "b.png"}, result
        server.invoke_agent = fake_agent

        # old path-keyed file is migrated, matching moved files by mtime
        moved = server.ROOT / "sub" / "moved" / "renamed.jpg"
        server.tags_path().write_text(json.dumps({"z0.jpg": {"mtime_ns": moved.stat().st_mtime_ns,
                                                             "tags": {"top": "old"}}}))
        assert server.stored_tags([moved]) == {moved: {"top": "old"}}
        assert "mtime_ns" not in server.tags_path().read_text()
        # image editing: pipeline and save never touch the original
        src = server.ROOT / "edit.png"
        Image.new("RGB", (200, 100), (100, 120, 140)).save(src)
        original = src.read_bytes()
        out = server.edit_image(src, {"crop": {"x": 50, "y": 0, "w": 100, "h": 100}, "size": {"w": 300, "h": 100}})
        assert out.size == (300, 100), out.size
        assert server.edit_image(src, {"size": {"w": 4000, "h": 2000}}, server.PREVIEW_SIDE).size == (1400, 700)
        mean = lambda img: img.convert("L").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        assert mean(server.edit_image(src, {"brightness": 2})) > mean(server.edit_image(src, {}))
        r, g, b = server.edit_image(src, {"temperature": 100}).getpixel((0, 0))
        assert r > 100 and b < 140, (r, g, b)
        assert server.edit_image(src, {"crop": {"x": -5, "y": 0, "w": 9999, "h": 9999}}).size == (200, 100)
        assert server.save_edited(src, {"brightness": 1.5}).name == "edit_edit.png"
        assert server.save_edited(src, {}).name == "edit_edit2.png"
        # editing an edited copy saves the original's next version, so it stays grouped under the original
        assert server.save_edited(src.with_name("edit_edit2.png"), {}).name == "edit_edit3.png"
        assert src.read_bytes() == original

        # auto enhance: model sees the cropped image + stats, out-of-range answers are clamped
        seen = {}

        def fake_enhance(job_dir, request):
            seen.update(request, size=Image.open(job_dir / request["references"][0]).size)
            return {"brightness": 1.2, "contrast": 9, "temperature": 20, "message": "m"}
        server.invoke_agent = fake_enhance
        result = server.auto_enhance(src, {"crop": {"x": 0, "y": 0, "w": 100, "h": 100}, "brightness": 1.8})
        assert seen["mode"] == "enhance" and seen["size"] == (100, 100) and 0 <= seen["stats"]["p50"] <= 255
        assert result["brightness"] == 1.2 and result["contrast"] == 3 and result["temperature"] == 20
        assert result["gamma"] == 1 and result["message"] == "m"

        # orientation: flip then rotate; crop coordinates are in the rotated space
        wide = server.ROOT / "wide.png"
        Image.new("RGB", (200, 100), (10, 20, 30)).save(wide)
        assert server.edit_image(wide, {"orient": {"rot": 90}}).size == (100, 200)
        assert server.edit_image(wide, {"orient": {"rot": 90}, "crop": {"x": 0, "y": 0, "w": 100, "h": 100}}).size == (100, 100)
        assert server.edit_image(wide, {"orient": {"rot": 77}}).size == (200, 100)  # invalid angle ignored

        # background removal is saved as PNG with alpha; face edits without a face fail clearly
        import numpy as np
        mask = np.zeros((100, 200), np.float32)
        mask[20:80, 60:140] = 1.0
        face = np.zeros((478, 3))
        server.detection = lambda *args: (mask, None)
        out = server.edit_image(wide, {"bg": {"mode": "remove"}})
        assert out.mode == "RGBA" and out.getchannel("A").getextrema() == (0, 255)
        jpg = server.ROOT / "photo.jpg"
        Image.new("RGB", (200, 100), "red").save(jpg)
        saved = server.save_edited(jpg, {"bg": {"mode": "remove"}})
        assert saved.name == "photo_edit.png" and Image.open(saved).mode == "RGBA", saved
        assert server.has_transparency(out) and not server.has_transparency(Image.new("RGBA", (4, 4), (0, 0, 0, 255)))
        assert server.checkerboard(out).mode == "RGB"
        try:
            server.edit_image(wide, {"face": {"brightness": 1.3}})
            raise AssertionError("face edit without a face")
        except ValueError as error:
            assert "얼굴" in str(error)
        blurred = server.edit_image(wide, {"bg": {"mode": "blur", "amount": 1.0}})
        assert blurred.mode == "RGB"

        # albums and duplicates go through the stored descriptions / files of a folder
        groups = server.album_groups("", "pose")
        assert groups["untagged"] >= 0 and all("label" in g for g in groups["groups"])
        assert isinstance(server.duplicate_groups("")["groups"], list)

        # rename plan: clusters asked once, tokens validated, sequences continue
        shop = server.ROOT / "shop"
        shop.mkdir()
        for i in range(2):
            Image.new("RGB", (90, 160), (0, 0, 200 + i)).save(shop / f"pending_{i}.png")
        for named in ("shop_redknit_fullbody_9x16_1k_01.png", "shop_redknit_fullbody_9x16_1k_01_edit2.png"):
            Image.new("RGB", (90, 160)).save(shop / named)  # already named (and its edit): left alone
        # undescribed photos are never analyzed without consent: library call and HTTP both refuse (409)
        calls.clear()
        try:
            server.rename_plan("shop", [])
            raise AssertionError("rename planned without consent")
        except server.NeedsIndex as error:
            assert error.count == 2 and calls == [], (error.count, calls)
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"

        def call(path, body=None):
            request = urllib.request.Request(base_url + path, data=None if body is None else json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())
        status, data = call("/api/rename/plan", {"dir": "shop"})
        assert status == 409 and data["needs_index"] == 2, (status, data)
        status, data = call("/api/agent", {"dir": "shop", "prompt": "앉은 자세"})
        assert status == 409 and data["needs_index"] == 4 and calls == [], (status, data, calls)  # search: every photo
        # bad query parameters answer 400 with a JSON error instead of dropping the connection
        status, data = call("/api/edit/base?path=wide.png&rot=abc")
        assert status == 400 and "error" in data, (status, data)
        status, data = call("/api/albums?dir=&kind=nonsense")
        assert status == 400 and "error" in data, (status, data)
        assert call("/api/dirs")[0] == 200
        httpd.shutdown()

        # merge_outfits: the richest photo's outfit goes to the others, originals are kept, unindexed photos refuse
        pair = [shop / n for n in ("pending_0.png", "pending_1.png")]
        saved = {pair[0]: {"visible": "fullbody", "top": "red shirt", "bottom": "blue jeans", "onepiece": "none"},
                 pair[1]: {"visible": "upperbody", "top": "crimson blouse", "outer": "none"}}
        stored = {}
        real_stored, real_store = server.stored_tags, server.store_tags
        server.stored_tags = lambda paths: {p: saved[p] for p in paths if p in saved}
        server.store_tags = stored.update
        try:
            result = server.merge_outfits([f"shop/{p.name}" for p in pair])
            assert result == {"count": 1, "base": "pending_0.png"}, result
            assert stored[pair[1]]["top"] == "red shirt" and stored[pair[1]]["bottom"] == "blue jeans"
            assert stored[pair[1]]["outfit_orig"]["top"] == "crimson blouse" and pair[0] not in stored
            saved.pop(pair[1])
            try:
                server.merge_outfits([f"shop/{p.name}" for p in pair])
                raise AssertionError("merged an unindexed photo")
            except ValueError:
                pass
        finally:
            server.stored_tags, server.store_tags = real_stored, real_store

        server.stored_tags = lambda paths: {p: {"top": "blue shirt", "bottom": "black skirt", "visible": "fullbody",
                                                "onepiece": "none", "outer": "none"} for p in paths}
        server.describe_images = lambda paths, status: None
        asked = {}

        def fake_names(job_dir, request):
            asked.update(request)
            return {"tokens": [{"id": "C1", "token": "Blue Shirt Black Skirt!"}]}
        server.invoke_agent = fake_names
        plan = server.rename_plan("shop", [])
        assert asked["mode"] == "outfit_token" and len(asked["clusters"]) == 1
        assert [p["new_name"] for p in plan["plan"]] == ["shop_blueshirtblackskirt_fullbody_9x16_1k_01.png",
                                                         "shop_blueshirtblackskirt_fullbody_9x16_1k_02.png"], plan
        done = server.rename_apply("shop", plan["plan"])
        assert done["count"] == 2 and (shop / plan["plan"][0]["new_name"]).exists()
        server.library.undo_op(server.ROOT, done["op"])
        assert (shop / "pending_0.png").exists()
        try:
            server.rename_apply("shop", [{"name": "../x.png", "new_name": "y.png"}])
            raise AssertionError("path in name")
        except ValueError:
            pass

        # rename_plan: different outfits that the model names alike get told apart; a manual name skips the model
        dark = {"visible": "fullbody", "onepiece": "dark floral print spaghetti strap mini dress", "top": "none", "bottom": "none", "outer": "none"}
        pale = {**dark, "onepiece": "beige floral sleeveless ruffle mini dress"}
        by_name = {"pending_0.png": dark, "pending_1.png": pale}
        server.stored_tags = lambda paths: {p: by_name[p.name] for p in paths if p.name in by_name}
        calls = []

        def same_name(job_dir, request):
            calls.append(request)
            return {"tokens": [{"id": c["id"], "token": "floralminidress"} for c in request["clusters"]]}
        server.invoke_agent = same_name
        plan = server.rename_plan("shop", ["pending_0.png", "pending_1.png"])["plan"]
        outfits = {p["name"]: p["outfit"] for p in plan}
        assert len(calls) == 2 and "floralminidress" in calls[1]["taken"], calls
        assert len(set(outfits.values())) == 2 and all("floralminidress" in o for o in outfits.values()), outfits
        calls.clear()
        plan = server.rename_plan("shop", ["pending_0.png", "pending_1.png"],
                                  overrides={"pending_0.png": "Black Pink Dress!", "pending_1.png": "blackpinkdress"})["plan"]
        assert not calls and {p["outfit"] for p in plan} == {"blackpinkdress"}, plan
        assert [p["new_name"][-6:] for p in plan] == ["01.png", "02.png"], plan
        try:
            server.rename_plan("shop", ["pending_0.png"], overrides={"pending_0.png": "x"})
            raise AssertionError("one-letter name accepted")
        except ValueError:
            pass

        # thumbnails: a cached one never waits for a lock, concurrent first requests produce one valid file
        cached = server.ROOT / "a.jpg"
        server.thumbnail(cached)
        fresh_source = server.ROOT / "thumb_new.png"
        Image.new("RGB", (900, 600), "blue").save(fresh_source)
        key = hashlib.sha256(b"thumb_new.png").hexdigest()
        stripe = server.THUMB_LOCKS[int(key[:8], 16) % len(server.THUMB_LOCKS)]
        with stripe:  # pretend another request is generating that thumbnail right now
            started = time.time()
            server.thumbnail(cached)
            assert time.time() - started < 1, "cached thumbnail waited for an unrelated lock"
        results = []
        workers = [threading.Thread(target=lambda: results.append(server.thumbnail(fresh_source))) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        assert len(set(results)) == 1 and max(Image.open(results[0]).size) <= server.THUMB_SIZE * 3
        assert not list(results[0].parent.glob(".*.tmp")), "temporary thumbnail files were left behind"

        # questions about selected photos go to the ask mode with just those photos, in selection order
        assert server.is_question("첫번째와 두번째 인물이 얼마나 유사한지 %로 판단해줘", ["a.jpg", "b.png"])
        assert not server.is_question("같은 의상 찾아줘", ["a.jpg"]) and not server.is_question("두 사진 비교해줘", [])
        asked_refs = {}
        def fake_ask(job_dir, request):
            asked_refs.update(request, files=sorted(p.name for p in job_dir.iterdir()))
            return {"observations": "x", "answer": "약 80% 유사합니다."}
        server.invoke_agent = fake_ask
        result = server.ask_images("", "얼마나 비슷해?", ["b.png", "a.jpg"])
        assert result == {"answer": "약 80% 유사합니다.", "images": ["b.png", "a.jpg"]}, result
        assert asked_refs["mode"] == "ask" and asked_refs["references"] == ["ref_01.jpg", "ref_02.jpg"]
        try:
            server.ask_images("", "q", ["a.jpg"] * 5)
            raise AssertionError("too many images")
        except ValueError:
            pass

        # best shot: scored locally (no agent call), the sharp one wins over its blurred copy
        import numpy as np
        noise = np.random.default_rng(3).integers(0, 255, (200, 160, 3), dtype=np.uint8)
        Image.fromarray(noise).save(server.ROOT / "shot_sharp.png")
        Image.fromarray(noise).filter(__import__("PIL.ImageFilter").ImageFilter.GaussianBlur(3)).save(server.ROOT / "shot_soft.png")
        server.invoke_agent = lambda *a: (_ for _ in ()).throw(AssertionError("best shot must not call the model"))
        picked = server.best_shot([server.ROOT / "shot_soft.png", server.ROOT / "shot_sharp.png"])
        assert picked["best"] == "shot_sharp.png" and len(picked["shots"]) == 2, picked
        try:
            server.best_shot([server.ROOT / "shot_sharp.png"])
            raise AssertionError("one photo is not a group")
        except ValueError:
            pass

        # face shape commands: dropped with an explanation when the face is missing or small, kept otherwise
        server.invoke_agent = lambda job_dir, request: {"nose_size": -12, "brightness": 1.1, "message": "코를 줄였어요"}
        server.face_points = lambda *key: None
        result = server.edit_command("코 작게", {}, [], cached, {})
        assert "nose_size" not in result and result["brightness"] == 1.1 and "얼굴" in result["message"], result
        import numpy as np
        face = np.zeros((478, 3))
        face[152, 1] = 300  # a 300px face in this small photo: large enough
        server.face_points = lambda *key: face
        assert server.edit_command("코 작게", {}, [], cached, {})["nose_size"] == -12
        assert server.shape_params({"shape": {"nose_size": -50, "jaw_width": 0, "x": 3}}) == {"nose_size": -20.0}

        # PWA bits
        assert json.loads(server.MANIFEST)["display"] == "standalone" and "fetch" in server.SERVICE_WORKER
        assert Image.open(io.BytesIO(server.app_icon(192))).size == (192, 192)
    print("ok")


if __name__ == "__main__":
    main()
