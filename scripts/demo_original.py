import cv2
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import List
import re
import warnings
import sys
# -----------------------------------------------------------------------------
# Configuration – tweak here if needed
# -----------------------------------------------------------------------------
TEMPERATURE = 0.1
TOP_P       = 0.15
DURATION_SEC = 5        # default recording length
FPS          = 20

# PROMPT 👉 one‑line action/state only, no appearance/background
PROMPT = (
    "Video: <image><image><image><image><image><image><image><image>\n"
    "Return **one concise English sentence** that describes ONLY the subject's action or state change. "
    "Do NOT mention appearance, colour, clothing, background, objects, or physical attributes."
)

# DAM script location — resolve project root one level above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .. (repo root)
DAM_SCRIPT   = PROJECT_ROOT / "src" / "dam_video_with_sam2.py"
if not DAM_SCRIPT.exists():
    raise FileNotFoundError(f"DAM script not found at: {DAM_SCRIPT}")

# I/O paths (under project root)
CAPTURE_DIR = PROJECT_ROOT / "captures"
CAPTURE_DIR.mkdir(exist_ok=True)
LOG_FILE    = PROJECT_ROOT / "action_log.txt"


# -----------------------------------------------------------------------------
# Helper: run DAM+SAM‑2 and return one‑line description (progress lines filtered)
# -----------------------------------------------------------------------------

def _extract_description(raw: str) -> str:
    """Strip tqdm/progress logs & warnings → return the Description line or last clean line."""
    desc = ""
    for line in raw.splitlines():
        if line.startswith("Description:"):
            desc = line.split("Description:", 1)[1].strip()
    if desc:
        return desc

    # fallback – pick the last non‑empty line that is not a progress bar/warning
    clean_lines = [l for l in raw.splitlines() if l.strip() and not re.search(r"frame loading|propagate in video|Loading checkpoint|UserWarning", l)]
    return clean_lines[-1].strip() if clean_lines else raw.strip()


def describe_video(video_path: Path, box_norm: List[float]) -> str:
    """Run the DAM+SAM‑2 CLI with fixed prompt → return one‑line description."""
    cmd = [
        sys.executable, str(DAM_SCRIPT),
        "--video_file", str(video_path),
        "--box", str(box_norm),
        "--normalized_coords",
        "--use_box",
        "--no_stream",
        "--temperature", str(TEMPERATURE),
        "--top_p",      str(TOP_P),
        "--query", PROMPT,
    ]

    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("[DAM stderr] ↓↓↓")
        print(result.stderr)
        raise RuntimeError(f"DAM exited {result.returncode}")

    return _extract_description(result.stdout or result.stderr)

# -----------------------------------------------------------------------------
# ROI selection on first frame – returns normalised box [x1,y1,x2,y2]
# -----------------------------------------------------------------------------

def select_roi(video_path: Path) -> List[float]:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame.")

    x, y, w, h = cv2.selectROI("Select ROI (Enter/Space = OK, ESC = Cancel)", frame, False, False)
    cv2.destroyWindow("Select ROI (Enter/Space = OK, ESC = Cancel)")

    if w == 0 or h == 0:  # user cancelled – use full frame
        return [0.0, 0.0, 1.0, 1.0]

    h_img, w_img = frame.shape[:2]
    box_norm = [x / w_img, y / h_img, (x + w) / w_img, (y + h) / h_img]
    return [round(v, 4) for v in box_norm]

# -----------------------------------------------------------------------------
# Logging helper – append to action_log.txt in two‑column TSV format
# -----------------------------------------------------------------------------

def append_log(start_dt: datetime, end_dt: datetime, description: str) -> None:
    """Append a row: <YYYY‑MM‑DD‑HHMMSS~HHMMSS> \t <description>"""
    time_range = f"{start_dt.strftime('%Y-%m-%d-%H%M%S')}~{end_dt.strftime('%H%M%S')}"
    with LOG_FILE.open("a", encoding="utf8") as f:
        f.write(f"{time_range}\t{description}\n")

# -----------------------------------------------------------------------------
# Record clip → run DAM in background thread, then log result
# -----------------------------------------------------------------------------

def record_and_describe(cap: cv2.VideoCapture, duration: int = DURATION_SEC, fps: int = FPS):
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_dt = datetime.now()
    vid_path = CAPTURE_DIR / f"video_{start_dt.strftime('%Y%m%d_%H%M%S')}.mp4"

    vw = cv2.VideoWriter(str(vid_path),
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    if not vw.isOpened():
        warnings.warn("VideoWriter failed to open – check codec/FourCC")
        return

    t0 = time.time()
    while time.time() - t0 < duration:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(frame)

    vw.release()
    print(f"[INFO] recording saved: {vid_path}")
    end_dt = start_dt + timedelta(seconds=duration)

    # run ROI selection + DAM asynchronously
    def _run():
        try:
            box_norm = select_roi(vid_path)
            desc     = describe_video(vid_path, box_norm)
            print(f"[DAM] {desc}")
            append_log(start_dt, end_dt, desc)
        except Exception as e:
            print("[ERR] DAM inference failed:", e)

    Thread(target=_run, daemon=True).start()

# -----------------------------------------------------------------------------
# Main camera loop – press 's' to record, 'q' to quit
# -----------------------------------------------------------------------------

def main():
    # 카메라 1 사용 (더 안정적)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("카메라 1을 열 수 없습니다. 카메라 0을 시도합니다...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return
    
    # 카메라 설정 최적화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 줄여서 지연 감소
    
    print("s: record 5 seconds | q: quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("[INFO] recording 5 seconds …")
            Thread(target=record_and_describe, args=(cap,), daemon=True).start()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 환경 체크 (선택) - 경고 메시지 숨김
    try:
        import sam2, torch
        # if not hasattr(sam2, "_C"):
        #     warnings.warn("⚠ SAM2 C-extension not found – using Dummy predictor (quality↓)")

        if not torch.cuda.is_available():
            warnings.warn("⚠ CUDA not available – inference will run on CPU (slow)")
    except ImportError:
        warnings.warn("sam2 or torch not importable – please check installation")

    main()
