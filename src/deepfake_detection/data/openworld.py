"""Open-world DF40 benchmark: every generator belongs to exactly one role.

Training uses DF40's FF++ training archives. Development and test use DF40's
evaluation archives on Celeb-DF identities and held-out FF++ clips, plus
collections made with pipelines DF40 never trains on. Those collections pair
real and fake images that differ in resolution and encoding (for example 1024 px
PNG fakes against 178 x 218 JPEG reals), so both classes of each collection pass
through one face-crop, scale-matching, and JPEG step before a model sees them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .df40 import COLUMNS as DF40_COLUMNS
from .identity import IDENTITY_TOKEN
from .materialize import _extract_one, _matches_existing

FS, FR, EFS, FE = "face_swap", "face_reenactment", "entire_face_synthesis", "face_edit"
SPLITS = ("train", "validation", "test")
PROTOCOL_V1 = "df40_openworld_v1"
PROTOCOL_ROUND2 = "df40_openworld_round2"
ROUND2_TRAIN_METHODS = {"kandinsky22", "sana600m", "recon_movq", "recon_dcae"}
ROUND2_VALIDATION_METHODS = {"stable_cascade_lite"}

# DF40 training archives contain FF++ training clips only.
TRAIN_METHODS = {
    "simswap": ("train/simswap.zip", FS),
    "blendface": ("train/blendface.zip", FS),
    "faceswap": ("train/faceswap.zip", FS),
    "fsgan": ("train/fsgan.zip", FS),
    "mobileswap": ("train/mobileswap.zip", FS),
    "e4s": ("train/e4s.zip", FS),
    "fomm": ("train/fomm.zip", FR),
    "facevid2vid": ("train/facevid2vid.zip", FR),
    "lia": ("train/lia.zip", FR),
    "tpsm": ("train/tpsm.zip", FR),
    "sadtalker": ("train/sadtalker.zip", FR),
    "wav2lip": ("train/wav2lip.zip", FR),
    "one_shot_free": ("train/one_shot_free.zip", FR),
    "stylegan2": ("train/StyleGAN2.zip", EFS),
    "styleganxl": ("train/StyleGANXL.zip", EFS),
    "sd21": ("train/sd2.1.zip", EFS),
    "dit": ("train/DiT.zip", EFS),
    "ddim": ("train/ddim.zip", EFS),
    "vqgan": ("train/VQGAN.zip", EFS),
}
# Held-out DF40 generators, read from the evaluation archives (Celeb-DF and FF++).
EVAL_METHODS = {
    "facedancer": ("validation", "test/facedancer.zip", FS),
    "mraa": ("validation", "test/MRAA.zip", FR),
    "danet": ("validation", "test/danet.zip", FR),
    "stylegan3": ("validation", "test/StyleGAN3.zip", EFS),
    "sit": ("validation", "test/SiT.zip", EFS),
    "uniface": ("test", "test/uniface.zip", FS),
    "inswap": ("test", "test/inswap.zip", FS),
    "mcnet": ("test", "test/mcnet.zip", FR),
    "hyperreenact": ("test", "test/hyperreenact.zip", FR),
    "pirender": ("test", "test/pirender.zip", FR),
    "rddm": ("test", "test/RDDM.zip", EFS),
    "pixart": ("test", "test/pixart.zip", EFS),
}
# Collections from other pipelines. Each one is its own domain with its own reals.
EXTERNAL = {
    "stargan": ("validation", "test/stargan.zip", FE),
    "styleclip": ("validation", "test/styleclip.zip", FE),
    "whichfaceisreal": ("validation", "test/whichfaceisreal.zip", EFS),
    "deepfacelab": ("test", "test/deepfacelab.zip", FS),
    "heygen": ("test", "test/heygen.zip", FR),
    "midjourney": ("test", "test/MidJourney.zip", EFS),
    "collabdiff": ("test", "test/CollabDiff.zip", FE),
    "starganv2": ("test", "test/starganv2.zip", FE),
}
# e4e inverts real FF++ frames; it is paired with the same clips' real frames.
E4E = ("test", "test/e4e.zip", FE)
REAL_ARCHIVES = {
    "ff": "real/FaceForensics++_real_data_for_DF40.zip",
    "cdf": "real/Celeb-DF-v2_real_data_for_DF40.zip",
}
# (videos per method and domain, frames per video); real frames per video.
SAMPLING = {
    "train": {"videos": 400, "frames": 4, "real_frames": 24},
    "validation": {"videos": 60, "frames": 4, "real_frames": 8},
    "test": {"videos": 150, "frames": 4, "real_frames": 8},
}
EXTERNAL_IMAGES_PER_CLASS = 1000
EXTERNAL_FRAMES_PER_VIDEO = 8
# Share of held-out FF++ clips and Celeb-DF identities assigned to development.
VALIDATION_SHARE = 0.4
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif")
FF_CLIPS = re.compile(r"^(\d{3})(?:_(\d{3}))?(?=_|$)")
FF_SOURCE_FRAME = re.compile(r"^(\d{3})_\d+$")
COLUMNS = (*DF40_COLUMNS, "role", "archive", "member", "preprocess")
CANONICAL_ID = "yunet-det320-arcface5-scale1.3-256-scalematch-jpeg90-v2"
# v3 applies the minimum face size at the source resolution rather than the 320 px
# detection scale; v1 (manifest adf10611e74ccef2191d) was built with v2.
CANONICAL_ID_ORIGINAL_SIZE = "yunet-det320-origsize-arcface5-scale1.3-256-scalematch-jpeg90-v3"
DETECTION_SIDE = 320
JPEG_QUALITY = 90


def method_roles(protocol: str = PROTOCOL_V1) -> dict[str, set[str]]:
    """Fake methods allowed in each split, for the lineage audit."""
    if protocol not in {PROTOCOL_V1, PROTOCOL_ROUND2}:
        raise ValueError(f"unknown open-world protocol: {protocol}")
    roles: dict[str, set[str]] = {"train": set(TRAIN_METHODS), "validation": set(), "test": set()}
    for method, (split, _, _) in (*EVAL_METHODS.items(), *EXTERNAL.items(), ("e4e", E4E)):
        roles[split].add(method)
    if protocol == PROTOCOL_ROUND2:
        roles["train"].update(ROUND2_TRAIN_METHODS)
        roles["validation"].update(ROUND2_VALIDATION_METHODS)
    return roles


def _rank(seed: str, *parts: str) -> str:
    return hashlib.sha256("\0".join((seed, *parts)).encode()).hexdigest()


def _image_members(archive: Path) -> list[str]:
    with ZipFile(archive) as stream:
        names = stream.namelist()
    return sorted(
        name for name in names
        if name.lower().endswith(IMAGE_SUFFIXES)
        and not any(part.startswith(("._", "__MACOSX")) for part in name.split("/"))
    )


def _safe(value: str, name: str) -> str:
    if not value or value in {".", ".."} or any(c in value for c in '/\\:<>"|?*\x00'):
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


def _frame_key(name: str) -> tuple:
    stem = PurePosixPath(name).stem
    digits = re.search(r"(\d+)$", stem)
    return (int(digits.group(1)) if digits else -1, stem)


def _even(items: list, limit: int) -> list:
    if len(items) <= limit:
        return list(items)
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)] if limit > 1 \
        else [items[len(items) // 2]]


def df40_member(member: str, *, default_domain: str | None) -> tuple[str, str, str] | None:
    """Return (domain, video, frame) for a DF40 archive member, or None if unusable.

    Celeb-DF's YouTube-real clips carry opaque names without identity evidence,
    so frames derived from them are left out, as in the frozen protocol.
    """
    parts = PurePosixPath(member).parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe archive member: {member}")
    if any("youtube-real" in part.casefold() for part in parts):
        return None
    domains = {part.casefold() for part in parts[:-2] if part.casefold() in {"ff", "cdf"}}
    if len(domains) > 1:
        raise ValueError(f"ambiguous domain: {member}")
    domain = domains.pop() if domains else default_domain
    if domain is None:
        raise ValueError(f"missing source domain: {member}")
    return domain, _safe(parts[-2], "video name"), _safe(parts[-1], "frame name")


def source_entities(domain: str, video: str, frame: str = "") -> tuple[str, ...]:
    """Identity units a sample derives from: FF++ source clips or Celeb-DF idNN tokens.

    Diffusion archives name each image after its conditioning frame (``006_199`` or
    ``Celeb-real_id1_0004_171``), so the file stem is lineage evidence as well.
    """
    stem = PurePosixPath(frame).stem
    if domain == "cdf":
        tokens = set(IDENTITY_TOKEN.findall(video)) | set(IDENTITY_TOKEN.findall(stem))
        return tuple(sorted(f"cdf:{token}" for token in tokens))
    clips = set()
    match = FF_CLIPS.match(video)
    if match:
        clips.update(value for value in match.groups() if value)
    source = FF_SOURCE_FRAME.match(stem)
    if source:
        clips.add(source.group(1))
    return tuple(sorted(f"ff:{clip}" for clip in clips))


class Ownership:
    """Assign every identity unit to one split before any sample is chosen."""

    def __init__(self, seed: str, train_entities: set[str], eval_entities: set[str]) -> None:
        self.owner: dict[str, str] = {}
        cutoff = int(VALIDATION_SHARE * 2**32)
        for entity in sorted(eval_entities):
            value = int(_rank(seed, "owner", entity)[:8], 16)
            self.owner[entity] = "validation" if value < cutoff else "test"
        # An evaluation identity never trains, even if a training archive reuses it.
        for entity in sorted(train_entities - eval_entities):
            self.owner[entity] = "train"

    def accepts(self, split: str, entities: tuple[str, ...]) -> bool:
        return bool(entities) and all(self.owner.get(entity) == split for entity in entities)


def _row(*, domain: str, method: str, family: str, split: str, video: str, frame: str,
         entities: tuple[str, ...], archive: str, member: str, preprocess: str,
         identity: str = "") -> dict[str, str]:
    label = "FAKE" if method else "REAL"
    suffix = ".jpg" if preprocess != "copy" else PurePosixPath(frame).suffix.lower()
    token = hashlib.sha256(f"{archive}\0{member}".encode()).hexdigest()[:12]
    relative = PurePosixPath(
        "images", domain, method or "real", split, video,
        f"{token}_{PurePosixPath(frame).stem}{suffix}",
    ).as_posix()
    return {
        "image_id": hashlib.sha256(relative.encode()).hexdigest()[:24],
        "relative_path": relative,
        "label": label,
        "split": split,
        "official_splits": "train" if archive.startswith("train/") else "test",
        "fake_method": method,
        "manipulation_family": family if method else "real",
        "source_domain": domain,
        "video_id": f"{domain}:{video}",
        "identity_id": identity,
        "frame_index": str(_frame_key(frame)[0]),
        "compression": "",
        "source_json": f"archive:{archive}",
        "source_path": member,
        "source_video_id": "",
        "target_identity_id": "",
        "lineage_video_ids": "|".join(entities),
        "role": split,
        "archive": archive,
        "member": member,
        "preprocess": preprocess,
    }


def _grouped(members: list[str], default_domain: str | None) -> dict[tuple[str, str], list]:
    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for member in members:
        parsed = df40_member(member, default_domain=default_domain)
        if parsed is None:
            continue
        domain, video, frame = parsed
        groups[(domain, video)].append((frame, member))
    return groups


def _sample_groups(seed: str, split: str, method: str, family: str, archive: str,
                   groups: dict[tuple[str, str], list], ownership: Ownership,
                   videos: int, frames: int) -> tuple[list[dict[str, str]], Counter]:
    rows: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    synthesis = family == EFS
    for domain in sorted({domain for domain, _ in groups}):
        eligible = []
        for (group_domain, video), items in groups.items():
            if group_domain != domain:
                continue
            if synthesis:
                # Synthesis folders borrow real video names without deriving from them,
                # and mix images conditioned on unrelated clips. Each image's
                # conditioning-frame name is the only identity evidence, so images
                # conditioned on another split's identities leave the folder.
                items = [(frame, member) for frame, member in items
                         if not source_entities(domain, "", frame)
                         or ownership.accepts(split, source_entities(domain, "", frame))]
                if not items:
                    skipped[f"{domain}:other_split"] += 1
                    continue
            entity_sets = {source_entities(domain, "" if synthesis else video, frame)
                           for frame, _ in items}
            entities = tuple(sorted({e for value in entity_sets for e in value}))
            identity = ""
            if synthesis:
                # Renaming the folder keeps its borrowed idNN token out of the audit.
                name = f"syn-{hashlib.sha256(f'{method}/{video}'.encode()).hexdigest()[:12]}"
                if entities:
                    identity = "|".join(e.removeprefix("cdf:") for e in entities
                                        if e.startswith("cdf:"))
                else:
                    # Generated from noise: a person owned by this generator alone.
                    entities = (f"synthetic:{method}:{domain}:{video}",)
                    identity = f"synthetic-{method}-{video}"
                    eligible.append((_rank(seed, split, method, domain, video), name, items,
                                     entities, identity))
                    continue
            else:
                name = video
            # Every frame of a group must belong to identities owned by this split.
            if not ownership.accepts(split, entities):
                skipped[f"{domain}:{'no_identity' if not entities else 'other_split'}"] += 1
                continue
            eligible.append((_rank(seed, split, method, domain, video), name, items, entities,
                             identity))
        for _, video, items, entities, identity in sorted(eligible)[:videos]:
            for frame, member in _even(sorted(items, key=lambda item: _frame_key(item[0])),
                                       frames):
                rows.append(_row(domain=domain, method=method, family=family, split=split,
                                 video=video, frame=frame, entities=entities, archive=archive,
                                 member=member, preprocess="copy", identity=identity))
    return rows, skipped


def _external_rows(name: str, split: str, family: str, archive: str, members: list[str],
                   seed: str) -> list[dict[str, str]]:
    """Label by the ``real``/``fake`` folder; video folders group frames when present."""
    by_class: dict[str, dict[str, list[tuple[str, str]]]] = {"REAL": {}, "FAKE": {}}
    for member in members:
        parts = PurePosixPath(member).parts
        lowered = [part.casefold() for part in parts]
        label = "FAKE" if "fake" in lowered else "REAL" if "real" in lowered else ""
        if not label:
            raise ValueError(f"cannot label external member: {member}")
        anchor = max(i for i, part in enumerate(lowered) if part in {"fake", "real", "frames"})
        # Folders below the class folder are clips or per-image folders.
        video = parts[-2] if len(parts) - anchor > 2 else PurePosixPath(parts[-1]).stem
        by_class[label].setdefault(_safe(video, "external video"), []).append(
            (parts[-1], member)
        )
    rows = []
    for label, videos in by_class.items():
        if not videos:
            raise ValueError(f"{name} has no {label} images")
        ranked = sorted(videos, key=lambda video: _rank(seed, name, label, video))
        budget = EXTERNAL_IMAGES_PER_CLASS
        for video in ranked:
            if budget <= 0:
                break
            items = sorted(videos[video], key=lambda item: _frame_key(item[0]))
            for frame, member in _even(items, min(EXTERNAL_FRAMES_PER_VIDEO, budget)):
                identity = f"{name}:{label.lower()}:{video}"
                rows.append(_row(domain=name, method=name if label == "FAKE" else "",
                                 family=family, split=split, video=f"{label.lower()}-{video}",
                                 frame=frame, entities=(identity,), archive=archive,
                                 member=member, preprocess=f"canonical:{name}",
                                 identity=identity))
                budget -= 1
    return rows


def build_catalog(downloads_root: Path, output: Path, *, seed: str) -> dict:
    """Write the selected rows of the open-world protocol as a DF40-style catalog."""
    train_groups = {
        method: _grouped(_image_members(downloads_root / archive), "ff")
        for method, (archive, _) in TRAIN_METHODS.items()
    }
    eval_groups = {
        method: _grouped(_image_members(downloads_root / archive), None)
        for method, (_, archive, _) in EVAL_METHODS.items()
    }
    real_groups = {
        domain: _grouped(_image_members(downloads_root / archive), domain)
        for domain, archive in REAL_ARCHIVES.items()
    }
    # Identity pools come from swap and reenactment archives, which are built from
    # real source clips. Diffusion conditioning frames reference clips across all of
    # FF++ and would otherwise claim training clips for evaluation.
    train_entities = {
        entity
        for method, groups in train_groups.items() if TRAIN_METHODS[method][1] != EFS
        for (domain, video), items in groups.items()
        for frame, _ in items
        for entity in source_entities(domain, video, frame)
    }
    eval_entities = {
        entity
        for method, groups in eval_groups.items() if EVAL_METHODS[method][2] != EFS
        for (domain, video), items in groups.items()
        for frame, _ in items
        for entity in source_entities(domain, video, frame)
    }
    # Celeb-DF identities never train, so all of them are evaluation units.
    eval_entities |= {
        entity
        for (domain, video), items in real_groups["cdf"].items()
        for entity in source_entities(domain, video)
    }
    ownership = Ownership(seed, train_entities, eval_entities)

    rows: list[dict[str, str]] = []
    skipped: dict[str, dict[str, int]] = {}
    for method, (archive, family) in TRAIN_METHODS.items():
        sample = SAMPLING["train"]
        selected, dropped = _sample_groups(seed, "train", method, family, archive,
                                           train_groups[method], ownership,
                                           sample["videos"], sample["frames"])
        rows += selected
        skipped[method] = dict(dropped)
    for method, (split, archive, family) in EVAL_METHODS.items():
        sample = SAMPLING[split]
        selected, dropped = _sample_groups(seed, split, method, family, archive,
                                           eval_groups[method], ownership,
                                           sample["videos"], sample["frames"])
        rows += selected
        skipped[method] = dict(dropped)
    # Real frames: every clip or identity owned by a split, at a fixed frame budget.
    used = {(row["split"], row["source_domain"]) for row in rows}
    for domain, groups in real_groups.items():
        for (_, video), items in sorted(groups.items()):
            entities = source_entities(domain, video)
            split = ownership.owner.get(entities[0]) if len(entities) == 1 else None
            if split is None or (split, domain) not in used:
                continue
            ordered = sorted(items, key=lambda item: _frame_key(item[0]))
            for frame, member in _even(ordered, SAMPLING[split]["real_frames"]):
                rows.append(_row(domain=domain, method="", family="", split=split, video=video,
                                 frame=frame, entities=entities,
                                 archive=REAL_ARCHIVES[domain], member=member,
                                 preprocess="copy"))
    for name, (split, archive, family) in EXTERNAL.items():
        rows += _external_rows(name, split, family, archive,
                               _image_members(downloads_root / archive), seed)
    rows += _e4e_rows(downloads_root, real_groups["ff"], ownership, seed)

    rows.sort(key=lambda row: row["relative_path"])
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("catalog produced duplicate destination paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((row["split"], row["source_domain"], row["fake_method"] or "real")
                     for row in rows)
    return {
        "catalog": str(output),
        "rows": len(rows),
        "seed": seed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "owned_entities": dict(sorted(Counter(ownership.owner.values()).items())),
        "skipped_groups": skipped,
    }


def _e4e_rows(downloads_root: Path, ff_real: dict, ownership: Ownership,
              seed: str) -> list[dict[str, str]]:
    """Pair e4e inversions of test-owned FF++ clips with those clips' real frames.

    Each FF++ folder holds the inversions of that clip's 32 frames in order, so
    the k-th inversion and the k-th real frame show the same moment.
    """
    split, archive, family = E4E
    inversions: dict[str, list[str]] = defaultdict(list)
    for member in _image_members(downloads_root / archive):
        parts = PurePosixPath(member).parts
        if len(parts) >= 3 and parts[-3] == "ff":
            inversions[_safe(parts[-2], "e4e clip")].append(member)
    clips = []
    for clip, members in inversions.items():
        entities = source_entities("ff", clip)
        real = ff_real.get(("ff", clip))
        if ownership.accepts(split, entities) and real and len(real) == len(members):
            clips.append((_rank(seed, "e4e", clip), clip, entities, members, real))
    rows = []
    for _, clip, entities, members, real in sorted(clips)[:SAMPLING[split]["videos"]]:
        fakes = sorted(members, key=_frame_key)
        reals = sorted(real, key=lambda item: _frame_key(item[0]))
        for position in _even(list(range(len(fakes))), EXTERNAL_FRAMES_PER_VIDEO):
            fake_member = fakes[position]
            real_frame, real_member = reals[position]
            rows.append(_row(domain="e4e", method="e4e", family=family, split=split,
                             video=f"fake-{clip}", frame=PurePosixPath(fake_member).name,
                             entities=entities, archive=archive, member=fake_member,
                             preprocess="canonical:e4e", identity="|".join(entities)))
            rows.append(_row(domain="e4e", method="", family=family, split=split,
                             video=f"real-{clip}", frame=real_frame, entities=entities,
                             archive=REAL_ARCHIVES["ff"], member=real_member,
                             preprocess="canonical:e4e", identity="|".join(entities)))
    return rows


# --- Materialization -------------------------------------------------------------

_DETECTOR = None


def _directory_member(downloads_root: str | Path, archive: str, member: str) -> Path:
    """Resolve ``member`` inside a ``dir:<folder>`` source, which holds plain files."""
    folder = (Path(downloads_root) / archive.removeprefix("dir:")).resolve()
    path = (folder / member).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"member escapes its source folder: {member}")
    return path


def _read_member(downloads_root: str, archive: str, member: str) -> bytes:
    if archive.startswith("dir:"):
        return _directory_member(downloads_root, archive, member).read_bytes()
    path = Path(downloads_root) / archive
    try:
        with ZipFile(path) as stream:
            return stream.read(member)
    except NotImplementedError:
        # Deflate64 members (the e4e archive) need Info-ZIP.
        return subprocess.run(["unzip", "-p", str(path), member], check=True,
                              capture_output=True).stdout


def _decode(data: bytes):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.seek(0)  # animated GIFs: first frame
        return np.asarray(image.convert("RGB"))


def _detect(job: tuple[str, str, str, bool]) -> dict:
    from deepfake_detection.inference import face

    global _DETECTOR
    if _DETECTOR is None:
        import cv2

        # One OpenCV thread per worker process; the pool supplies the parallelism.
        cv2.setNumThreads(1)
        _DETECTOR = face.FaceDetector()
    downloads_root, archive, member, size_at_original = job
    try:
        rgb = _decode(_read_member(downloads_root, archive, member))
    except (OSError, ValueError):
        return {"reason": "unreadable_image"}
    # YuNet misses many faces that fill a 1024 px frame; it finds them reliably at
    # this scale. Landmarks are mapped back to full resolution for alignment.
    scale = min(1.0, DETECTION_SIDE / max(rgb.shape[:2]))
    if scale < 1.0:
        import cv2

        size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    if size_at_original:
        # The inference policy (largest confident face, minimum short side), with
        # the size limit applied in source pixels so downscaling excludes nothing.
        faces = _DETECTOR.detect(rgb)
        confident = [found for found in faces if found.score >= face.MIN_SCORE]
        if not faces:
            return {"reason": "no_face"}
        if not confident:
            return {"reason": "low_confidence"}
        if confident[0].short_side / scale < face.MIN_SHORT_SIDE:
            return {"reason": "face_too_small"}
        points = confident[0].landmarks / scale
    else:
        result = face.extract_face(rgb, _DETECTOR)
        if result.crop is None:
            return {"reason": result.reason}
        points = result.face.landmarks / scale
    return {"reason": "", "landmarks": points.tolist(),
            "eyes": float(((points[0] - points[1]) ** 2).sum() ** 0.5)}


def _canonical(job: tuple[str, str, str, list, float, str]) -> int:
    """Downscale so the eye distance is at most ``target``, align, and JPEG-encode."""
    import cv2
    import numpy as np
    from PIL import Image

    from deepfake_detection.inference import face

    cv2.setNumThreads(1)
    downloads_root, archive, member, landmarks, target, destination = job
    rgb = _decode(_read_member(downloads_root, archive, member))
    points = np.asarray(landmarks, dtype=np.float64)
    eyes = float(np.linalg.norm(points[0] - points[1]))
    scale = min(1.0, target / eyes)
    if scale < 1.0:
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        points = points * scale
    crop = face.align(rgb, points)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    Image.fromarray(crop).save(temporary, format="JPEG", quality=JPEG_QUALITY,
                               subsampling="4:2:0")
    temporary.replace(path)
    return path.stat().st_size


def materialize(catalog: Path, downloads_root: Path, data_root: Path, output: Path, *,
                workers: int = 8, size_at_original: bool = False) -> dict:
    """Write every catalog image under ``data_root`` and the surviving rows to ``output``.

    DF40 crops are copied byte for byte. External collections are canonicalized
    per collection: the target eye distance is the smallest median among its real
    images and each fake method, so no class keeps a resolution advantage.
    Sources named ``dir:<folder>`` hold plain files instead of a ZIP archive.
    """
    with catalog.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    root = data_root.resolve()

    def destination(row: dict[str, str]) -> Path:
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root) or PurePosixPath(row["relative_path"]).parts[0] \
                != "images":
            raise ValueError(f"unsafe destination: {row['relative_path']}")
        return path

    copied = reused = 0
    by_archive: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["preprocess"] == "copy":
            by_archive[row["archive"]].append(row)
    for archive, items in sorted(by_archive.items()):
        if archive.startswith("dir:"):
            for row in items:
                source = _directory_member(downloads_root, archive, row["member"])
                target = destination(row)
                if target.exists():
                    if target.read_bytes() != source.read_bytes():
                        raise RuntimeError(f"refusing to overwrite changed image: {target}")
                    reused += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".part")
                temporary.write_bytes(source.read_bytes())
                temporary.replace(target)
                copied += 1
            continue
        with ZipFile(downloads_root / archive) as stream:
            for row in items:
                member = stream.getinfo(row["member"])
                target = destination(row)
                if target.exists():
                    if not _matches_existing(target, member):
                        raise RuntimeError(f"refusing to overwrite changed image: {target}")
                    reused += 1
                    continue
                _extract_one(stream, member, target)
                copied += 1

    canonical = [row for row in rows if row["preprocess"] != "copy"]
    jobs = [(str(downloads_root), row["archive"], row["member"], size_at_original)
            for row in canonical]
    with ProcessPoolExecutor(workers) as pool:
        detections = list(pool.map(_detect, jobs, chunksize=16))
    collections: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, found in zip(canonical, detections, strict=True):
        if not found["reason"]:
            collections[row["preprocess"]][row["fake_method"] or "real"].append(found["eyes"])
    targets = {}
    for name, classes in collections.items():
        if "real" not in classes or len(classes) < 2:
            raise ValueError(f"{name}: real or fake images have no detected faces")
        targets[name] = min(statistics.median(values) for values in classes.values())
    canonical_jobs = []
    excluded: Counter[str] = Counter()
    kept = {id(row) for row in rows if row["preprocess"] == "copy"}
    for row, found in zip(canonical, detections, strict=True):
        if found["reason"]:
            excluded[f"{row['preprocess']}/{row['label']}/{found['reason']}"] += 1
            continue
        kept.add(id(row))
        canonical_jobs.append((str(downloads_root), row["archive"], row["member"],
                               found["landmarks"], targets[row["preprocess"]],
                               str(destination(row))))
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_canonical, canonical_jobs, chunksize=16))
    surviving = [row for row in rows if id(row) in kept]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(surviving)
    selected = Counter(f"{row['preprocess']}/{row['label']}" for row in canonical)
    return {
        "catalog": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(surviving),
        "copied": copied,
        "reused": reused,
        "canonicalized": len(canonical_jobs),
        "canonical_id": CANONICAL_ID_ORIGINAL_SIZE if size_at_original else CANONICAL_ID,
        "target_eye_distance": {name: round(value, 2) for name, value in targets.items()},
        "selected_canonical": dict(sorted(selected.items())),
        "excluded": dict(sorted(excluded.items())),
    }


def prune(catalog: Path, manifest_dir: Path, output: Path) -> dict:
    """Remove lower-priority video groups whose images collide across splits.

    DF40 itself ships some StyleGAN3 files byte-identical to StyleGAN2 files, and
    several external collections draw reals from CelebA, so exact and perceptual
    collisions are resolved here with the frozen protocol's rule.
    """
    from .manifest import verify_manifest
    from .prune import FINGERPRINTS, collision_removals

    valid, errors = verify_manifest(manifest_dir)
    if not valid:
        raise ValueError(f"invalid manifest: {errors}")
    with (manifest_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    with catalog.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if {row["image_id"] for row in rows} != {row["source_id"] for row in manifest}:
        raise ValueError("manifest does not describe this catalog")
    remove, collisions = collision_removals(manifest)
    kept = [row for row in rows if (row["split"], row["video_id"]) not in remove]
    fingerprints = {row["source_id"]: row for row in manifest}
    for column in FINGERPRINTS:
        owners: dict[str, set[str]] = defaultdict(set)
        for row in kept:
            owners[fingerprints[row["image_id"]][column]].add(row["split"])
        if any(len(splits) > 1 for splits in owners.values()):
            raise RuntimeError(f"cross-split {column} collisions remain")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    removed = Counter(f"{row['split']}/{row['source_domain']}/{row['fake_method'] or 'real'}"
                      for row in rows if (row["split"], row["video_id"]) in remove)
    return {
        "catalog": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(kept),
        "collision_groups": dict(collisions),
        "removed_rows": dict(sorted(removed.items())),
        "source_manifest_id": manifest_dir.name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--downloads-root", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--seed", required=True)
    build = commands.add_parser("materialize")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--downloads-root", type=Path, required=True)
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--size-at-original-resolution", action="store_true",
                       help="apply the minimum face size in source pixels (fresh v2)")
    deduplicate = commands.add_parser("prune")
    deduplicate.add_argument("--catalog", type=Path, required=True)
    deduplicate.add_argument("--manifest-dir", type=Path, required=True)
    deduplicate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prune":
        result = prune(args.catalog, args.manifest_dir, args.output)
    elif args.command == "catalog":
        result = build_catalog(args.downloads_root, args.output, seed=args.seed)
    else:
        result = materialize(args.catalog, args.downloads_root, args.data_root, args.output,
                             workers=args.workers,
                             size_at_original=args.size_at_original_resolution)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
