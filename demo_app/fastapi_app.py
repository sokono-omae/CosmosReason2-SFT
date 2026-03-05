"""CosmosReason2 FastAPI app (self-contained).

Streams video frames over WebSocket and matches the display FPS to the video FPS.
"""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime
import os
import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Full
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont


def _first_nonempty_env(*names: str, default: str = "") -> str:
    for name in names:
        val = os.environ.get(name)
        if val is None:
            continue
        val = str(val).strip()
        if val:
            return val
    return default


# ---------------------------------------------------------------------------
# Inference worker and HUD (ported from app.py)
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    text: str
    latency_ms: float
    video_time_sec: float
    request_epoch_sec: float
    error: Optional[str] = None
    encode_ms: Optional[float] = None
    api_ms: Optional[float] = None


class CosmosReason2Worker(threading.Thread):
    def __init__(
        self,
        api_base_url: str | None,
        api_key: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        jpeg_quality: int,
    ) -> None:
        super().__init__(daemon=True)
        client_kwargs = {"api_key": api_key}
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.jpeg_quality = jpeg_quality
        self._jobs: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=1)
        self._latest_result: Optional[InferenceResult] = None
        self._result_lock = threading.Lock()
        self._busy = False
        self._busy_lock = threading.Lock()

    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._busy

    def submit(self, frame_bgr: np.ndarray, video_time_sec: float) -> bool:
        if self.is_busy():
            return False
        while not self._jobs.empty():
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        self._jobs.put((frame_bgr.copy(), video_time_sec))
        return True

    def get_latest(self) -> Optional[InferenceResult]:
        with self._result_lock:
            return self._latest_result

    def run(self) -> None:
        while True:
            frame_bgr, video_time_sec = self._jobs.get()
            with self._busy_lock:
                self._busy = True
            request_start = time.time()
            try:
                text, encode_ms, api_ms = self._infer_one(frame_bgr, video_time_sec)
                result = InferenceResult(
                    text=text,
                    latency_ms=(time.time() - request_start) * 1000.0,
                    video_time_sec=video_time_sec,
                    request_epoch_sec=request_start,
                    encode_ms=encode_ms,
                    api_ms=api_ms,
                )
            except Exception as exc:
                result = InferenceResult(
                    text="",
                    latency_ms=(time.time() - request_start) * 1000.0,
                    video_time_sec=video_time_sec,
                    request_epoch_sec=request_start,
                    error=str(exc),
                )
            finally:
                with self._busy_lock:
                    self._busy = False
            with self._result_lock:
                self._latest_result = result

    def _infer_one(self, frame_bgr: np.ndarray, video_time_sec: float) -> tuple[str, Optional[float], Optional[float]]:
        t0 = time.perf_counter()
        ok, jpeg_buf = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed.")
        image_b64 = base64.b64encode(jpeg_buf).decode("ascii")
        encode_ms = (time.perf_counter() - t0) * 1000.0
        prompt = self.user_prompt.replace("{video_time:.2f}", f"{video_time_sec:.2f}").replace(
            "{video_time}", str(video_time_sec)
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        t1 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        api_ms = (time.perf_counter() - t1) * 1000.0
        content = response.choices[0].message.content
        return (content.strip() if content else "(empty response)", encode_ms, api_ms)


def _font_candidates() -> list[Path]:
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    fonts_dir = windows_dir / "Fonts"
    return [
        fonts_dir / "msgothic.ttc",
        fonts_dir / "meiryo.ttc",
        fonts_dir / "YuGothR.ttc",
        fonts_dir / "yugothic.ttc",
        fonts_dir / "arial.ttf",
    ]


def load_japanese_font(font_path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        explicit = Path(font_path)
        if explicit.exists():
            return ImageFont.truetype(str(explicit), size=size)
        raise FileNotFoundError(f"font file not found: {font_path}")
    for candidate in _font_candidates():
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _font_with_size(base_font: ImageFont.ImageFont, size: int) -> ImageFont.ImageFont:
    if hasattr(base_font, "font_variant"):
        try:
            return base_font.font_variant(size=max(8, int(size)))
        except Exception:
            return base_font
    return base_font


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _fit_single_line_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    base_font: ImageFont.ImageFont,
    max_width_px: int,
    preferred_size: int,
    min_size: int = 10,
) -> tuple[str, ImageFont.ImageFont]:
    text = " ".join(text.split())
    if not text:
        return "", _font_with_size(base_font, preferred_size)
    font = _font_with_size(base_font, preferred_size)
    if hasattr(base_font, "font_variant"):
        for size in range(preferred_size, min_size - 1, -1):
            cand_font = _font_with_size(base_font, size)
            if _text_width(draw, text, cand_font) <= max_width_px:
                return text, cand_font
        font = _font_with_size(base_font, min_size)
    if _text_width(draw, text, font) > max_width_px:
        ellipsis = "..."
        if _text_width(draw, ellipsis, font) > max_width_px:
            return "", font
        while text and _text_width(draw, text + ellipsis, font) > max_width_px:
            text = text[:-1]
        text = (text + ellipsis) if text else ellipsis
    return text, font


def _parse_result_lines(text: str) -> list[str]:
    """Extract action_name/part/slot/action from inference text as one-line strings."""
    text = text.strip()
    if not text:
        return []
    lines: list[str] = []
    for key in ("action_name", "part", "slot", "action"):
        m = re.search(rf'["\']?{re.escape(key)}["\']?\s*[:\=;]\s*["\']?([^"\',\]\}}]+)["\']?', text, re.I)
        if m:
            val = m.group(1).strip()
            lines.append(f"{key}: {val}")
    if lines:
        return lines
    if "\n" in text:
        return [ln.strip() for ln in text.split("\n") if ln.strip()][:4]
    return [text]


def _extract_action_name_from_text(text: str) -> Optional[str]:
    """Extract a single action_name value from inference text."""
    if not text or not text.strip():
        return None
    m = re.search(
        r'["\']?action_name["\']?\s*[:\=;]\s*["\']?([^"\',\]\}}\s]+)["\']?',
        text.strip(),
        re.I,
    )
    return m.group(1).strip() if m else None


def _extract_action_id_from_text(text: str) -> Optional[int]:
    """Extract a single action_id (0-5) from inference text.

    If the VLM returns an explicit action_id, prefer it.
    """
    if not text or not text.strip():
        return None
    m = re.search(
        r'["\']?action_id["\']?\s*[:\=;]\s*(\d+)',
        text.strip(),
        re.I,
    )
    if not m:
        return None
    try:
        v = int(m.group(1))
        if 0 <= v <= 5:
            return v
    except (ValueError, IndexError):
        pass
    return None


VLM_ACTION_WINDOW = 10


def _most_likely_action(vlm_history: list[dict], window: int = VLM_ACTION_WINDOW) -> tuple[Optional[str], int]:
    """Return the most frequent action_name in the last `window` VLM history entries."""
    if not vlm_history or window <= 0:
        return (None, 0)
    recent = vlm_history[-window:]
    counts: dict[str, int] = {}
    for row in recent:
        text = row.get("text") or ""
        name = _extract_action_name_from_text(text)
        if name and name.strip():
            key = name.strip()
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return (None, 0)
    best_action = max(counts, key=counts.get)
    return (best_action, counts[best_action])


ACTION_STEP_ORDER = [
    "install_slot1_fan", "install_slot2_fan", "install_slot1_cover",
    "install_slot2_cover", "install_nic",
]
# Map action_name (new format) returned by the VLM to step indices
ACTION_NAME_TO_STEP = {
    "install_fan_1": 1,
    "install_fan_2": 2,
    "install_fan_cover_1": 3,
    "install_fan_cover_2": 4,
    "install_nic": 5,
}
CHART_DATA_MAX_POINTS = 2000


def _action_to_step(action_name: Optional[str]) -> int:
    if not action_name or not action_name.strip():
        return 0
    key = action_name.strip()
    # Treat no_action / non_action as step 0 (below the chart Y-axis)
    if key in ("no_action", "non_action"):
        return 0
    # Prefer the new-format action_name (install_fan_1, etc.)
    if key in ACTION_NAME_TO_STEP:
        return ACTION_NAME_TO_STEP[key]
    for i, a in enumerate(ACTION_STEP_ORDER, 1):
        if a == key:
            return i
    return 0


def _text_to_step(text: str) -> int:
    """Convert VLM inference text into a step index (0-6).

    Prefer action_id if present; otherwise convert from action_name.
    """
    aid = _extract_action_id_from_text(text)
    if aid is not None:
        return aid
    name = _extract_action_name_from_text(text)
    return _action_to_step(name)


def _build_chart_data(vlm_history: list[dict], max_points: int = CHART_DATA_MAX_POINTS) -> list[dict]:
    if not vlm_history:
        return []
    recent = vlm_history[-max_points:] if len(vlm_history) > max_points else vlm_history
    return [
        {"t": round(float(row.get("video_time_sec", 0)), 2), "step": _text_to_step(row.get("text") or ""), "action": (_extract_action_name_from_text(row.get("text") or "") or "")}
        for row in recent
    ]


MIN_STEP_DURATION_SEC = 1.5


def _steps_without_sufficient_duration(chart_data: list[dict], min_duration_sec: float = MIN_STEP_DURATION_SEC) -> list[int]:
    """Return step numbers that never sustain a segment >= min_duration_sec.

    step0 (no_action) is excluded.
    """
    steps_needed = set(range(1, len(ACTION_STEP_ORDER) + 1))
    if not chart_data:
        return sorted(steps_needed)
    valid = sorted([p for p in chart_data if p.get("step") in steps_needed], key=lambda p: float(p["t"]))
    if not valid:
        return list(steps_needed)
    for step in list(steps_needed):
        run_start_t: Optional[float] = None
        max_dur = 0.0
        for p in valid:
            t, s = float(p["t"]), p["step"]
            if s == step:
                if run_start_t is None:
                    run_start_t = t
                max_dur = max(max_dur, t - run_start_t)
            else:
                run_start_t = None
        if max_dur >= min_duration_sec:
            steps_needed.discard(step)
    return sorted(steps_needed)


def draw_hud(
    frame: np.ndarray,
    latest_result: Optional[InferenceResult],
    worker_busy: bool,
    source_desc: str,
    request_interval_sec: float,
    app_fps: float,
    title_font: ImageFont.ImageFont,
    meta_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    current_action: Optional[str] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    margin = 20
    panel_x, panel_y = margin, margin
    panel_w = max(180, (w // 2) - (margin * 2))
    panel_h = max(120, (h // 2) - (margin * 2))
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (24, 24, 32), -1)
    alpha = 0.58
    frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (90, 230, 255), 2)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    text_left = panel_x + 16
    text_width = panel_w - 32
    line_gap = max(4, int(panel_h * 0.03))
    y = panel_y + 12

    title_txt, title_fit_font = _fit_single_line_text(
        draw=draw, text="CosmosReason2 live overlay", base_font=title_font,
        max_width_px=text_width, preferred_size=max(14, int(panel_h * 0.12)), min_size=10,
    )
    draw.text((text_left, y), title_txt, font=title_fit_font, fill=(255, 255, 255))
    y += (title_fit_font.getbbox("Ag")[3] - title_fit_font.getbbox("Ag")[1]) + line_gap

    status_txt, status_fit_font = _fit_single_line_text(
        draw=draw, text=f"Display FPS: {app_fps:.1f}", base_font=meta_font,
        max_width_px=text_width, preferred_size=max(12, int(panel_h * 0.10)), min_size=10,
    )
    status_color = (255, 210, 0) if worker_busy else (120, 255, 120)
    draw.text((text_left, y), status_txt, font=status_fit_font, fill=status_color)
    y += (status_fit_font.getbbox("Ag")[3] - status_fit_font.getbbox("Ag")[1]) + line_gap

    if latest_result is None:
        wait_txt, wait_fit_font = _fit_single_line_text(
            draw=draw, text="Waiting for the first inference result...", base_font=body_font,
            max_width_px=text_width, preferred_size=max(11, int(panel_h * 0.09)), min_size=9,
        )
        draw.text((text_left, y), wait_txt, font=wait_fit_font, fill=(220, 220, 220))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    meta_text = f"Video Time = {latest_result.video_time_sec:.2f}s  Inference Latency = {latest_result.latency_ms:.1f} ms"
    if latest_result.error:
        err_head_txt, err_head_font = _fit_single_line_text(
            draw=draw, text="Last Request: Error", base_font=body_font,
            max_width_px=text_width, preferred_size=max(12, int(panel_h * 0.10)), min_size=10,
        )
        draw.text((text_left, y), err_head_txt, font=err_head_font, fill=(255, 80, 20))
        y += (err_head_font.getbbox("Ag")[3] - err_head_font.getbbox("Ag")[1]) + line_gap
        err_txt, err_font = _fit_single_line_text(
            draw=draw, text=latest_result.error, base_font=meta_font,
            max_width_px=text_width, preferred_size=max(10, int(panel_h * 0.08)), min_size=8,
        )
        draw.text((text_left, y), err_txt, font=err_font, fill=(255, 200, 200))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    meta_txt, meta_fit_font = _fit_single_line_text(
        draw=draw, text=meta_text, base_font=meta_font,
        max_width_px=text_width, preferred_size=max(10, int(panel_h * 0.08)), min_size=8,
    )
    draw.text((text_left, y), meta_txt, font=meta_fit_font, fill=(160, 240, 255))
    y += (meta_fit_font.getbbox("Ag")[3] - meta_fit_font.getbbox("Ag")[1]) + line_gap

    result_lines = _parse_result_lines(latest_result.text)
    line_font_size = max(14, int(panel_h * 0.11))
    for line in result_lines[:4]:
        line_txt, line_font = _fit_single_line_text(
            draw=draw, text=line, base_font=body_font,
            max_width_px=text_width, preferred_size=line_font_size, min_size=11,
        )
        if line_txt:
            draw.text((text_left, y), line_txt, font=line_font, fill=(255, 255, 255))
        y += (line_font.getbbox("Ag")[3] - line_font.getbbox("Ag")[1]) + line_gap
    if current_action:

        action_txt, action_font = _fit_single_line_text(
            draw=draw, text=f"expected action: {current_action}", base_font=body_font,
            max_width_px=text_width, preferred_size=max(11, int(panel_h * 0.09)), min_size=9,
        )
        draw.text((text_left, y), action_txt, font=action_font, fill=(180, 255, 180))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
VIDEO_DIR = APP_DIR / "video"
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "cosmos_reason2_uploads"
EXPORTS_DIR = APP_DIR / "exports"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

DEFAULT_SYSTEM_PROMPT = """
You are a strict video action classifier for server assembly.

Task:
- Analyze the provided video clip and identify the single current action.

Valid classes (closed set):
- 0: non_action
- 1: install_fan_1
- 2: install_fan_2
- 3: install_fan_cover_1
- 4: install_fan_cover_2
- 5: install_nic

Decision policy:
- Output exactly one class from the closed set above.
- If no installation step is actively being performed, output non_action (id=0).
- If motion is ambiguous, choose the most visually dominant ongoing action.
- Keep action_id and action_name strictly consistent with the mapping above.

Output format requirements:
- Return ONLY one JSON object.
- Use EXACTLY these keys in this order:
  {"action_id":0,"action_name":"non_action"}
- action_id must be an integer in [0,1,2,3,4,5].
- action_name must be exactly one of:
    non_action, install_fan_1, install_fan_2, install_fan_cover_1, install_fan_cover_2, install_nic
- No extra keys, no markdown, no explanation, no trailing text.
""".strip()

DEFAULT_USER_PROMPT = (
    "Determine the current server assembly status. Respond with JSON only using this exact schema: "
    '{"action_id":0,"action_name":"non_action"}'
    "Do not output any extra text."
)


def _load_fonts():
    try:
        title = load_japanese_font("", 24)
        meta = load_japanese_font("", 18)
        body = load_japanese_font("", 20)
        return title, meta, body
    except Exception:
        f = ImageFont.load_default()
        return f, f, f


def _list_videos() -> list[dict]:
    if not VIDEO_DIR.is_dir():
        return []
    return [
        {"path": str(p), "name": p.name}
        for p in sorted(VIDEO_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]


def _resolve_video_path(path: str) -> Path:
    p = Path(path).resolve()
    try:
        p.relative_to(VIDEO_DIR.resolve())
        return p
    except ValueError:
        pass
    try:
        p.relative_to(UPLOAD_DIR.resolve())
        return p
    except ValueError:
        raise HTTPException(400, "Invalid video_path")


def _run_inference_collect_history(
    path: Path,
    api_base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    jpeg_quality: int,
    request_interval_sec: float,
) -> list[dict]:
    """Play a video from the beginning and collect VLM results at request_interval_sec.

    Returns vlm_history.
    """
    worker = CosmosReason2Worker(
        api_base_url=api_base_url,
        api_key=api_key,
        model_name=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        jpeg_quality=jpeg_quality,
    )
    worker.start()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    vlm_history: list[dict] = []
    last_added_epoch: Optional[float] = None
    last_request_ts = 0.0
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or not (0.1 < video_fps < 120):
            video_fps = 30.0
        display_interval = 1.0 / video_fps
        while True:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if now - last_request_ts >= request_interval_sec:
                if worker.submit(frame, video_time_sec):
                    last_request_ts = now
            latest = worker.get_latest()
            if latest:
                is_new_result = (
                    last_added_epoch is None or latest.request_epoch_sec != last_added_epoch
                )
                if is_new_result and not latest.error and latest.text.strip():
                    vlm_history.append({
                        "text": latest.text,
                        "video_time_sec": latest.video_time_sec,
                        "latency_ms": latest.latency_ms,
                    })
                    last_added_epoch = latest.request_epoch_sec
            elapsed = time.perf_counter() - frame_start
            sleep_time = display_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        time.sleep(2.0)
        latest = worker.get_latest()
        if latest and not latest.error and latest.text.strip():
            if last_added_epoch is None or latest.request_epoch_sec != last_added_epoch:
                vlm_history.append({
                    "text": latest.text,
                    "video_time_sec": latest.video_time_sec,
                    "latency_ms": latest.latency_ms,
                })
    finally:
        cap.release()
    return vlm_history



def _csv_escape(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s)
    if not s:
        return ""
    if any(c in s for c in '",\r\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


def _build_report_csv(vlm_history: list[dict]) -> str:
    """Build a CSV string from VLM history."""
    lines = ["type,video_time_sec,content,latency_ms"]
    for row in vlm_history:
        t = row.get("video_time_sec", "")
        text = row.get("text", "")
        lat = row.get("latency_ms", "")
        lines.append(f"VLM,{t},{_csv_escape(text)},{lat}")
    return "\r\n".join(lines)


FONTS = _load_fonts()

app = FastAPI(title="CosmosReason2")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/videos")
async def api_list_videos():
    return {"videos": _list_videos()}


@app.get("/api/defaults")
async def api_defaults():
    return {
        "api_base_url": _first_nonempty_env("BASE_URL"),
        "api_key": os.environ.get("API_KEY", "EMPTY"),
        "model": _first_nonempty_env("MODEL"),
        "max_tokens": int(os.environ.get("MAX_TOKENS", "256")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.05")),
        "jpeg_quality": int(os.environ.get("JPEG_QUALITY", "88")),
        "request_interval_sec": float(os.environ.get("REQUEST_INTERVAL_SEC", "0.3")),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
        "min_step_duration_sec": float(os.environ.get("MIN_STEP_DURATION_SEC", str(MIN_STEP_DURATION_SEC))),
    }
    

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")
    path = UPLOAD_DIR / file.filename
    content = await file.read()
    path.write_bytes(content)
    return {"path": str(path), "name": path.name}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "message": "Config timeout"})
        await websocket.close()
        return

    video_path = config.get("video_path")
    if not video_path:
        await websocket.send_json({"type": "error", "message": "video_path required"})
        await websocket.close()
        return

    try:
        path = _resolve_video_path(video_path)
    except HTTPException as e:
        await websocket.send_json({"type": "error", "message": e.detail})
        await websocket.close()
        return

    if not path.exists():
        await websocket.send_json({"type": "error", "message": "File not found"})
        await websocket.close()
        return

    def _get(key: str, default, conv=str):
        v = config.get(key)
        if v is None or v == "":
            return default
        try:
            return conv(v)
        except (TypeError, ValueError):
            return default

    env_api = _first_nonempty_env("BASE_URL")
    env_key = os.environ.get("API_KEY", "EMPTY")
    env_model = _first_nonempty_env("MODEL")
    api_base_url = _get("api_base_url", env_api)
    api_key = _get("api_key", env_key)
    model = _get("model", env_model)

    if not model:
        await websocket.send_json(
            {
                "type": "error",
                "message": "model is required (set in UI or env MODEL)",
            }
        )
        await websocket.close()
        return

    max_tokens = int(_get("max_tokens", os.environ.get("MAX_TOKENS", "256")))
    temperature = float(_get("temperature", os.environ.get("TEMPERATURE", "0.05")))
    jpeg_quality = int(_get("jpeg_quality", os.environ.get("JPEG_QUALITY", "88")))
    request_interval_sec = float(_get("request_interval_sec", os.environ.get("REQUEST_INTERVAL_SEC", "0.3")))
    system_prompt = _get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    user_prompt = _get("user_prompt", DEFAULT_USER_PROMPT)
    min_step_duration_sec = float(_get("min_step_duration_sec", MIN_STEP_DURATION_SEC, lambda v: float(v)))

    queue: Queue = Queue(maxsize=1)
    overlay_input_queue: Queue = Queue(maxsize=1)
    _overlay_stop = object()

    def overlay_worker():
        """Overlay worker.

        Runs HUD rendering, JPEG encoding, and Base64 encoding in a separate thread, and
        pushes the result into the queue.
        """
        title_font, meta_font, body_font = FONTS
        while True:
            item = overlay_input_queue.get()
            if item is _overlay_stop:
                break
            (
                frame,
                result_data,
                latest,
                worker_busy,
                source_desc,
                req_interval_sec,
                app_fps,
                current_action,
                video_time_sec,
            ) = item
            vis = draw_hud(
                frame=frame,
                latest_result=latest,
                worker_busy=worker_busy,
                source_desc=source_desc,
                request_interval_sec=req_interval_sec,
                app_fps=app_fps,
                title_font=title_font,
                meta_font=meta_font,
                body_font=body_font,
                current_action=current_action,
            )
            _, jpeg_buf = cv2.imencode(".jpg", vis)
            b64 = base64.b64encode(jpeg_buf).decode("ascii")
            frame_msg = {
                "type": "frame",
                "image": f"data:image/jpeg;base64,{b64}",
                "result": result_data,
                "video_time_sec": video_time_sec,
                "fps": app_fps,
            }
            try:
                queue.put_nowait(frame_msg)
            except Full:
                try:
                    queue.get_nowait()
                except Exception:
                    pass
                try:
                    queue.put_nowait(frame_msg)
                except Full:
                    pass

    def run_video_loop():
        worker = CosmosReason2Worker(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            jpeg_quality=jpeg_quality,
        )
        worker.start()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            queue.put({"type": "error", "message": "Failed to open video"})
            return
        overlay_thread = threading.Thread(target=overlay_worker, daemon=True)
        overlay_thread.start()
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if not video_fps or not (0.1 < video_fps < 120):
                video_fps = 30.0
            display_interval = 1.0 / video_fps
            title_font, meta_font, body_font = FONTS
            source_desc = f"video:{path.name}"
            last_request_ts = 0.0
            vlm_history: list[dict] = []
            last_added_epoch: Optional[float] = None
            while True:
                frame_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                current_action: Optional[str] = None
                now = time.time()
                video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if now - last_request_ts >= request_interval_sec:
                    if worker.submit(frame, video_time_sec):
                        last_request_ts = now
                latest = worker.get_latest()
                result_data = None
                if latest:
                    result_data = {
                        "text": latest.text,
                        "latency_ms": latest.latency_ms,
                        "video_time_sec": latest.video_time_sec,
                        "error": latest.error,
                    }
                    is_new_result = (
                        last_added_epoch is None or latest.request_epoch_sec != last_added_epoch
                    )
                    if is_new_result and not latest.error and latest.text.strip():
                        vlm_history.append({
                            "text": latest.text,
                            "video_time_sec": latest.video_time_sec,
                            "latency_ms": latest.latency_ms,
                        })
                        last_added_epoch = latest.request_epoch_sec
                    current_action, current_action_count = _most_likely_action(vlm_history)
                    if result_data is not None:
                        result_data["current_action"] = current_action
                        result_data["current_action_count"] = current_action_count
                payload = (
                    frame.copy(),
                    result_data,
                    latest,
                    worker.is_busy(),
                    source_desc,
                    request_interval_sec,
                    video_fps,
                    current_action if latest else None,
                    video_time_sec,
                )
                try:
                    overlay_input_queue.put_nowait(payload)
                except Full:
                    try:
                        overlay_input_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        overlay_input_queue.put_nowait(payload)
                    except Full:
                        pass
                elapsed = time.perf_counter() - frame_start
                sleep_time = display_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            overlay_input_queue.put(_overlay_stop)
            overlay_thread.join(timeout=5.0)
            chart_data = _build_chart_data(vlm_history)
            step_errors = _steps_without_sufficient_duration(chart_data, min_step_duration_sec)
            queue.put({
                "type": "done",
                "vlm_history": vlm_history,
                "step_errors": step_errors,
                "min_step_duration_sec": min_step_duration_sec,
            })
        finally:
            cap.release()

    thread = threading.Thread(target=run_video_loop, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(
                    loop.run_in_executor(None, queue.get),
                    timeout=300.0,
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "message": "Timeout"})
                break
            await websocket.send_json(msg)
            if msg.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    await websocket.close()
