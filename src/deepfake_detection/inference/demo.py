"""Interactive image demonstration for a frozen deepfake detection bundle.

The app uses the same face selection, alignment, model, calibration, and decision
threshold as ``deepfake-detect``. It adds presentation only: bounding boxes, a
terminal-style calculation panel, and an exportable PNG snapshot.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import bundle, face, media

WINDOW_BACKGROUND = "#080d12"
PANEL_BACKGROUND = "#0c141b"
PANEL_FOREGROUND = "#b7f7cf"
MUTED = "#82909b"
REAL_COLOR = "#2ee59d"
FAKE_COLOR = "#ff5d73"
ABSTAIN_COLOR = "#ffbd59"
OTHER_FACE_COLOR = "#5aa9e6"
IMAGE_FILETYPES = [
    ("Images", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
]


@dataclass(frozen=True)
class ImageAnalysis:
    """All presentation data for one image, including the original RGB pixels."""

    path: Path
    rgb: np.ndarray
    faces: list[face.Face]
    selected: face.Face | None
    accepted: bool
    reason: str | None
    decision: dict[str, Any]
    frame_logit: float | None
    raw_probability: float | None
    calibrated_logit: float | None
    model_version: str
    threshold: float
    calibration_a: float
    calibration_b: float
    device: str
    detection_ms: float
    inference_ms: float
    total_ms: float


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def analyze_image(path: Path, scorer: bundle.Scorer, detector: face.FaceDetector) -> ImageAnalysis:
    """Read and score one image while retaining detector and timing details for display."""
    started = time.perf_counter()
    rgb = media.read_image(path)

    detection_started = time.perf_counter()
    faces = detector.detect(rgb)
    selected = face.select_face(rgb, faces)
    detection_ms = (time.perf_counter() - detection_started) * 1000

    inference_started = time.perf_counter()
    logits = scorer.frame_logits([selected.crop]) if selected.crop is not None else []
    decision = scorer.decide(logits, video=False)
    inference_ms = (time.perf_counter() - inference_started) * 1000

    frame_logit = float(logits[0]) if logits else None
    calibration = scorer.bundle["calibration"]
    calibration_a = float(calibration["a"])
    calibration_b = float(calibration["b"])
    calibrated_logit = (
        calibration_a * frame_logit + calibration_b if frame_logit is not None else None
    )
    device = str(getattr(scorer, "device", "unknown"))
    return ImageAnalysis(
        path=path,
        rgb=rgb,
        faces=faces,
        selected=selected.face,
        accepted=selected.crop is not None,
        reason=selected.reason,
        decision=decision,
        frame_logit=frame_logit,
        raw_probability=_sigmoid(frame_logit) if frame_logit is not None else None,
        calibrated_logit=calibrated_logit,
        model_version=str(scorer.bundle["model_version"]),
        threshold=float(scorer.bundle["threshold"]),
        calibration_a=calibration_a,
        calibration_b=calibration_b,
        device=device,
        detection_ms=detection_ms,
        inference_ms=inference_ms,
        total_ms=(time.perf_counter() - started) * 1000,
    )


def _selected_color(analysis: ImageAnalysis) -> str:
    label = analysis.decision["label"]
    if label == "FAKE":
        return FAKE_COLOR
    if label == "REAL":
        return REAL_COLOR
    return ABSTAIN_COLOR


def _is_selected(candidate: face.Face, selected: face.Face | None) -> bool:
    return selected is not None and candidate is selected


def annotated_image(analysis: ImageAnalysis) -> Image.Image:
    """Draw detector boxes and the selected-face decision on a copy of the input."""
    image = Image.fromarray(analysis.rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 180))
    font = _font(max(13, round(min(image.size) / 38)))

    for index, found in enumerate(analysis.faces, start=1):
        x, y, width, height = found.box
        box = (round(x), round(y), round(x + width), round(y + height))
        chosen = _is_selected(found, analysis.selected)
        color = _selected_color(analysis) if chosen else OTHER_FACE_COLOR
        draw.rectangle(box, outline=color, width=line_width)
        tag = f"FACE {index}  detector={found.score:.3f}"
        if chosen:
            probability = analysis.decision.get("probability_fake")
            if probability is not None:
                tag += f"  {analysis.decision['label']} {probability:.1%}"
            else:
                tag += f"  {analysis.decision['label']}"
        _label(draw, box[0], box[1], tag, font, color)
    return image


def _label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
           font: ImageFont.ImageFont, color: str) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0] + 12
    height = bounds[3] - bounds[1] + 8
    top = max(0, y - height)
    draw.rectangle((x, top, x + width, top + height), fill=color)
    draw.text((x + 6, top + 3), text, font=font, fill="#071017")


def _value(value: float | None, digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def terminal_lines(analysis: ImageAnalysis) -> list[tuple[str, str]]:
    """Return ``(text, style)`` rows for the on-screen diagnostics terminal."""
    height, width = analysis.rgb.shape[:2]
    selected = analysis.selected
    probability = analysis.decision.get("probability_fake")
    label = str(analysis.decision["label"])
    rows: list[tuple[str, str]] = [
        ("DEEPFAKE // IMAGE INSPECTOR", "heading"),
        ("", "normal"),
        ("[INPUT]", "section"),
        (f"file           {analysis.path.name}", "normal"),
        (f"resolution     {width} x {height}", "normal"),
        ("", "normal"),
        ("[FACE PIPELINE]", "section"),
        (f"detections     {len(analysis.faces)}", "normal"),
    ]
    if selected is None:
        rows.append(("selected       none", "normal"))
    else:
        x, y, box_width, box_height = selected.box
        rows.extend([
            (f"selected bb    x={x:.0f} y={y:.0f} w={box_width:.0f} h={box_height:.0f}",
             "normal"),
            (f"detector score {selected.score:.4f}", "normal"),
        ])
    rows.extend([
        (f"crop status    {'accepted' if analysis.accepted else analysis.reason}", "normal"),
        ("", "normal"),
        ("[MODEL CALCULATION]", "section"),
        (f"frame logit    {_value(analysis.frame_logit)}", "normal"),
        (f"raw sigmoid    {_value(analysis.raw_probability)}", "normal"),
        (f"Platt a, b     {analysis.calibration_a:.5f}, {analysis.calibration_b:.5f}",
         "normal"),
        (f"calibrated z   {_value(analysis.calibrated_logit)}", "normal"),
        (f"P(fake)        {_value(probability)}", "normal"),
        (f"threshold      {analysis.threshold:.5f}", "normal"),
        ("", "normal"),
        (f"> VERDICT      {label}", label.lower()),
        ("", "normal"),
        ("[RUNTIME]", "section"),
        (f"face pipeline  {analysis.detection_ms:8.1f} ms", "normal"),
        (f"model          {analysis.inference_ms:8.1f} ms", "normal"),
        (f"total          {analysis.total_ms:8.1f} ms", "normal"),
        (f"device         {analysis.device}", "normal"),
        ("", "normal"),
        ("[PROVENANCE]", "section"),
        (f"model          {analysis.model_version}", "normal"),
        (f"face policy    {face.POLICY_ID}", "normal"),
        ("", "normal"),
        ("Score is a calibrated model signal,", "muted"),
        ("not proof of image provenance.", "muted"),
    ])
    return rows


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["DejaVuSansMono-Bold.ttf", "Menlo Bold.ttf", "consolab.ttf"] if bold else
        ["DejaVuSansMono.ttf", "Menlo.ttc", "consola.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_snapshot(analysis: ImageAnalysis, *, max_image_size: tuple[int, int] = (960, 800)) \
        -> Image.Image:
    """Create a portable image containing the annotated input and calculation terminal."""
    annotated = ImageOps.contain(annotated_image(analysis), max_image_size)
    panel_width = 520
    line_height = 22
    padding = 24
    rows = terminal_lines(analysis)
    height = max(annotated.height, padding * 2 + line_height * len(rows))
    output = Image.new("RGB", (annotated.width + panel_width, height), WINDOW_BACKGROUND)
    output.paste(annotated, (0, (height - annotated.height) // 2))

    draw = ImageDraw.Draw(output)
    draw.rectangle((annotated.width, 0, output.width, height), fill=PANEL_BACKGROUND)
    normal = _font(15)
    bold = _font(16, bold=True)
    y = padding
    colors = {
        "heading": "#f4f7f9", "section": "#67d5ff", "normal": PANEL_FOREGROUND,
        "muted": MUTED, "real": REAL_COLOR, "fake": FAKE_COLOR,
        "abstain": ABSTAIN_COLOR,
    }
    for text, style in rows:
        draw.text((annotated.width + padding, y), text, font=bold if style in {
            "heading", "section", "real", "fake", "abstain"
        } else normal, fill=colors[style])
        y += line_height
    return output


class DemoWindow:
    """Small Tk viewer that can repeatedly ingest images with one loaded model."""

    def __init__(self, scorer: bundle.Scorer, detector: face.FaceDetector,
                 initial_paths: list[Path]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self.scorer = scorer
        self.detector = detector
        self.paths = initial_paths
        self.index = 0
        self.analysis: ImageAnalysis | None = None
        self.photo: Any = None

        self.root = tk.Tk()
        self.root.title("Deepfake Image Inspector")
        self.root.configure(background=WINDOW_BACKGROUND)
        self.root.geometry("1320x820")
        self.root.minsize(980, 620)
        self._build()
        if self.paths:
            self.root.after(50, self._load_current)
        else:
            self.root.after(100, self.open_images)

    def _build(self) -> None:
        tk, ttk = self._tk, self._ttk
        style = ttk.Style(self.root)
        style.configure("Demo.TFrame", background=WINDOW_BACKGROUND)
        style.configure("Demo.TButton", padding=(12, 7))

        toolbar = ttk.Frame(self.root, style="Demo.TFrame", padding=(12, 10))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open images", command=self.open_images,
                   style="Demo.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Previous", command=lambda: self._move(-1),
                   style="Demo.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Next", command=lambda: self._move(1),
                   style="Demo.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="Save snapshot", command=self.save_snapshot,
                   style="Demo.TButton").pack(side="left")
        self.status = tk.StringVar(value="Choose one or more images")
        tk.Label(toolbar, textvariable=self.status, background=WINDOW_BACKGROUND,
                 foreground="#d5dde3", font=("TkDefaultFont", 10)).pack(side="right")

        content = ttk.Frame(self.root, style="Demo.TFrame", padding=(12, 0, 12, 12))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        self.image_label = tk.Label(content, background="#05080b", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.terminal = tk.Text(
            content, background=PANEL_BACKGROUND, foreground=PANEL_FOREGROUND,
            insertbackground=PANEL_FOREGROUND, borderwidth=0, padx=20, pady=18,
            font=("TkFixedFont", 11), wrap="word", state="disabled", width=51,
        )
        self.terminal.grid(row=0, column=1, sticky="nsew")
        self.terminal.tag_configure("heading", foreground="#f4f7f9",
                                    font=("TkFixedFont", 12, "bold"))
        self.terminal.tag_configure("section", foreground="#67d5ff")
        self.terminal.tag_configure("muted", foreground=MUTED)
        self.terminal.tag_configure("real", foreground=REAL_COLOR,
                                    font=("TkFixedFont", 12, "bold"))
        self.terminal.tag_configure("fake", foreground=FAKE_COLOR,
                                    font=("TkFixedFont", 12, "bold"))
        self.terminal.tag_configure("abstain", foreground=ABSTAIN_COLOR,
                                    font=("TkFixedFont", 12, "bold"))
        self.root.bind("<Left>", lambda _event: self._move(-1))
        self.root.bind("<Right>", lambda _event: self._move(1))
        self.root.bind("<Control-o>", lambda _event: self.open_images())

    def open_images(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilenames(title="Choose demonstration images",
                                               filetypes=IMAGE_FILETYPES)
        if selected:
            self.paths = [Path(path) for path in selected]
            self.index = 0
            self._load_current()

    def _move(self, step: int) -> None:
        if not self.paths:
            return
        self.index = (self.index + step) % len(self.paths)
        self._load_current()

    def _load_current(self) -> None:
        from tkinter import messagebox

        from PIL import ImageTk

        path = self.paths[self.index]
        self.status.set(f"Analyzing {path.name} ...")
        self.root.update_idletasks()
        try:
            analysis = analyze_image(path, self.scorer, self.detector)
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set(f"Could not analyze {path.name}")
            messagebox.showerror("Deepfake Image Inspector", str(error))
            return
        self.analysis = analysis
        display = ImageOps.contain(annotated_image(analysis), (800, 740))
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo)
        self._write_terminal(terminal_lines(analysis))
        self.status.set(f"{self.index + 1} / {len(self.paths)}  |  {path.name}")

    def _write_terminal(self, rows: list[tuple[str, str]]) -> None:
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        for text, style in rows:
            self.terminal.insert("end", text + "\n", style)
        self.terminal.configure(state="disabled")

    def save_snapshot(self) -> None:
        from tkinter import filedialog

        if self.analysis is None:
            return
        default = self.analysis.path.with_name(self.analysis.path.stem + "-deepfake-inspection.png")
        destination = filedialog.asksaveasfilename(
            title="Save inspection snapshot", initialfile=default.name,
            defaultextension=".png", filetypes=[("PNG image", "*.png")],
        )
        if destination:
            render_snapshot(self.analysis).save(destination)
            self.status.set(f"Saved {Path(destination).name}")

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Release bundle directory containing bundle.json and best.pt")
    parser.add_argument("--device", help="Torch device, for example cpu, cuda, or mps")
    parser.add_argument("--detector-model", type=Path,
                        help="Optional local copy of the pinned YuNet ONNX file")
    parser.add_argument("--snapshot", type=Path,
                        help="Write one annotated PNG without opening a window")
    parser.add_argument("paths", type=Path, nargs="*", help="Initial image files")
    args = parser.parse_args(argv)
    if args.snapshot is not None and len(args.paths) != 1:
        parser.error("--snapshot requires exactly one image path")

    scorer = bundle.Scorer(args.bundle, device=args.device)
    detector = face.FaceDetector(args.detector_model)
    if args.snapshot is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        render_snapshot(analyze_image(args.paths[0], scorer, detector)).save(args.snapshot)
        print(args.snapshot)
        return 0
    try:
        DemoWindow(scorer, detector, args.paths).run()
    except ImportError as error:
        raise RuntimeError(
            "Tkinter is required for the demo window; use --snapshot for headless output"
        ) from error
    return 0


if __name__ == "__main__":
    sys.exit(main())
