#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path


MAGIC = "QRTX1"

# Bytes of compressed payload carried by each QR frame.
# 650 is conservative enough for reliable real-time scanning.
DEFAULT_CHUNK_SIZE = 650

# Start conservatively. Increase to 5-8 FPS if the camera is stable.
DEFAULT_FPS = 4.0


# ============================================================
# Utilities
# ============================================================

def require_runtime_deps():
    try:
        import cv2          # noqa: F401
        import numpy        # noqa: F401
        import qrcode       # noqa: F401
    except ImportError as e:
        print(
            "Missing dependency.\n"
            "Install with:\n\n"
            "    python -m pip install 'qrcode[pil]' opencv-python numpy\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from e


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_filename(name: str) -> str:
    return base64.urlsafe_b64encode(name.encode("utf-8")).decode("ascii")


def decode_filename(encoded: str) -> str:
    raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
    return raw.decode("utf-8", errors="replace")


# ============================================================
# Protocol / encoding
# ============================================================

def build_payloads(path: Path, chunk_size: int) -> tuple[list[str], dict]:
    raw = path.read_bytes()
    compressed = zlib.compress(raw, level=9)
    sha = file_sha256(raw)

    sid = f"{sha[:8]}-{random.randrange(0, 65536):04x}"
    total = max(1, math.ceil(len(compressed) / chunk_size))

    meta = {
        "m": MAGIC,
        "s": sid,
        "n": total,
        "f": encode_filename(path.name),
        "o": len(raw),
        "z": len(compressed),
        "h": sha,
    }

    payloads: list[str] = []

    for i in range(total):
        chunk = compressed[i * chunk_size:(i + 1) * chunk_size]

        frame = {
            **meta,
            "i": i,
            "c": f"{zlib.crc32(chunk) & 0xffffffff:08x}",
            "d": base64.b85encode(chunk).decode("ascii"),
        }

        payloads.append(
            json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
        )

    return payloads, meta


def parse_frame(text: str) -> tuple[dict, bytes] | None:
    try:
        obj = json.loads(text)

        if obj.get("m") != MAGIC:
            return None

        required = {"s", "n", "f", "o", "z", "h", "i", "c", "d"}
        if not required.issubset(obj):
            return None

        n = int(obj["n"])
        i = int(obj["i"])

        if n <= 0 or not (0 <= i < n):
            return None

        chunk = base64.b85decode(obj["d"].encode("ascii"))
        crc = f"{zlib.crc32(chunk) & 0xffffffff:08x}"

        if crc.lower() != str(obj["c"]).lower():
            return None

        return obj, chunk

    except Exception:
        return None


def same_session(meta: dict, obj: dict) -> bool:
    keys = ("s", "n", "f", "o", "z", "h")
    return all(meta.get(k) == obj.get(k) for k in keys)


# ============================================================
# QR generation
# ============================================================

def payload_to_qr(payload: str):
    import cv2
    import numpy as np
    import qrcode

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=4,
    )

    qr.add_data(payload)
    qr.make(fit=True)

    pil = qr.make_image(
        fill_color="black",
        back_color="white",
    ).convert("L")

    gray = np.array(pil)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_sender_status(img, text1: str, text2: str):
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    band = 82

    canvas = np.full((h + band, w, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = img

    cv2.putText(
        canvas,
        text1,
        (16, h + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text2,
        (16, h + 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# Sender
# ============================================================

def send_file(path: Path, fps: float, chunk_size: int):
    import cv2

    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    if fps <= 0:
        raise SystemExit("--fps must be > 0")

    if not 100 <= chunk_size <= 1200:
        raise SystemExit("--chunk-size must be between 100 and 1200 bytes")

    payloads, meta = build_payloads(path, chunk_size)

    total = len(payloads)
    delay_ms = max(1, int(1000 / fps))
    original = meta["o"]
    compressed = meta["z"]

    print()
    print("QRTX sender")
    print("=" * 50)
    print(f"File:       {path.name}")
    print(f"Original:   {original:,} bytes")
    print(f"Compressed: {compressed:,} bytes")
    print(f"Frames:     {total}")
    print(f"Session:    {meta['s']}")
    print()
    print("Press Q or Esc in the QR window to stop.")

    cv2.namedWindow("QRTX Sender", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QRTX Sender", 900, 980)

    order = list(range(total))
    cycle = 0

    try:
        while True:
            if cycle > 0:
                random.shuffle(order)

            for idx in order:
                qr = payload_to_qr(payloads[idx])

                shown = add_sender_status(
                    qr,
                    f"Frame {idx + 1}/{total}   cycle {cycle + 1}",
                    f"session {meta['s']}   {fps:g} FPS   compressed {compressed:,} B",
                )

                cv2.imshow("QRTX Sender", shown)
                key = cv2.waitKey(delay_ms) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    return

            cycle += 1

    finally:
        cv2.destroyAllWindows()


# ============================================================
# Receiver output helpers
# ============================================================

def choose_output_path(out_dir: Path, filename: str) -> Path:
    safe_name = Path(filename).name or "received.bin"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = out_dir / safe_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix

    for i in range(1, 10000):
        alt = out_dir / f"{stem}_received_{i}{suffix}"
        if not alt.exists():
            return alt

    raise RuntimeError("Could not choose a free output filename")


def finalize_received(meta: dict, chunks: dict[int, bytes], out_dir: Path) -> Path:
    total = int(meta["n"])
    compressed = b"".join(chunks[i] for i in range(total))

    expected_compressed = int(meta["z"])
    if len(compressed) != expected_compressed:
        raise ValueError(
            "Compressed size mismatch: "
            f"expected {expected_compressed}, got {len(compressed)}"
        )

    raw = zlib.decompress(compressed)

    expected_original = int(meta["o"])
    if len(raw) != expected_original:
        raise ValueError(
            "Original size mismatch: "
            f"expected {expected_original}, got {len(raw)}"
        )

    actual_sha = file_sha256(raw)
    if actual_sha.lower() != str(meta["h"]).lower():
        raise ValueError("SHA-256 verification failed")

    filename = decode_filename(str(meta["f"]))
    out_path = choose_output_path(out_dir, filename)
    temp_path = out_path.with_name(out_path.name + ".part")

    temp_path.write_bytes(raw)
    temp_path.replace(out_path)

    return out_path


# ============================================================
# Receiver UI
# ============================================================

def draw_receiver_status(
    frame,
    received: int,
    total: int | None,
    session: str | None,
):
    import cv2

    if total:
        pct = 100.0 * received / total
        line1 = f"QRTX: {received}/{total}  ({pct:.1f}%)"
    else:
        line1 = "QRTX: waiting for first frame..."

    line2 = f"session: {session or '-'}   Q/Esc to quit"

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 74),
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        frame,
        line1,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        line2,
        (12, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Receiver
# ============================================================

def receive_file(camera: int, out_dir: Path, width: int, height: int):
    import cv2

    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = cv2.QRCodeDetector()

    meta: dict | None = None
    chunks: dict[int, bytes] = {}
    last_new = 0.0

    print()
    print("QRTX receiver")
    print("=" * 50)
    print("Point the camera at the sender screen.")
    print("Receiver will lock onto the first QRTX session.")
    print("Press Q or Esc to quit.")

    cv2.namedWindow("QRTX Receiver", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QRTX Receiver", 1000, 700)

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                continue

            text, points, _ = detector.detectAndDecode(frame)

            if text:
                parsed = parse_frame(text)

                if parsed is not None:
                    obj, chunk = parsed

                    if meta is None:
                        meta = {
                            k: obj[k]
                            for k in ("s", "n", "f", "o", "z", "h")
                        }

                        print(f"Locked session {meta['s']}")
                        print(f"Frames: {meta['n']}")
                        print(f"Compressed: {meta['z']:,} bytes")

                    if same_session(meta, obj):
                        idx = int(obj["i"])

                        if idx not in chunks:
                            chunks[idx] = chunk
                            last_new = time.time()
                            total = int(meta["n"])

                            print(
                                f"\rReceived {len(chunks)}/{total}",
                                end="",
                                flush=True,
                            )

                        if points is not None:
                            pts = points.astype(int).reshape(-1, 2)
                            cv2.polylines(frame, [pts], True, (0, 180, 0), 3)

                        if len(chunks) == int(meta["n"]):
                            print()
                            print("All frames received.")
                            print("Verifying...")

                            out_path = finalize_received(meta, chunks, out_dir)

                            print()
                            print("SHA-256 verified:")
                            print(meta["h"])
                            print()
                            print("Saved:")
                            print(out_path.resolve())
                            return

            total = int(meta["n"]) if meta else None
            session = meta.get("s") if meta else None

            draw_receiver_status(
                frame,
                len(chunks),
                total,
                session,
            )

            if last_new and meta and (time.time() - last_new > 5):
                cv2.putText(
                    frame,
                    "No new frame for 5s: move closer / refocus / lower sender FPS",
                    (12, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("QRTX Receiver", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                return

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qrtx",
        description="Offline file transfer using animated QR codes.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("send", help="Show a file as a looping QR stream")
    ps.add_argument("file", type=Path)
    ps.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ps.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    pr = sub.add_parser("receive", help="Receive a QR stream from a camera")
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--out", type=Path, default=Path("received"))
    pr.add_argument("--width", type=int, default=1920)
    pr.add_argument("--height", type=int, default=1080)

    return p


def main():
    require_runtime_deps()
    args = build_parser().parse_args()

    if args.cmd == "send":
        send_file(args.file, args.fps, args.chunk_size)
    elif args.cmd == "receive":
        receive_file(args.camera, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
