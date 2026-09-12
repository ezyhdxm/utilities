#!/usr/bin/env python3
"""Experimental qrtx extensions. Same protocol, riskier settings.

Everything here trades reliability for speed and needs testing on your
actual screen/camera pair before you trust it:

- send:      like qrtx send, but allows chunk sizes up to 2800 bytes
             (QR version 40). Dense codes need the QR large in the
             camera frame; run calibration first.
- caltx /    calibration: caltx cycles test codes of increasing chunk
  calrx:     size on the sender screen; calrx watches them and reports
             the decode rate per size, recommending the largest size
             your setup decodes reliably.
- csend /    color multiplexing: three independent protocol frames per
  creceive:  displayed image, one per RGB channel (~3x throughput).
             Highly sensitive to screen color profile and camera white
             balance.

Requires qrtx.py next to this file. Both machines must run the same
protocol version; the wire format is unchanged.
"""
from __future__ import annotations

import argparse
import os
import random
import time
import zlib
from pathlib import Path

import qrtx

LAB_MAX_CHUNK = 2800  # ~QR version 40 at ECC L in our encoding

CAL_MAGIC = "QLC"
CAL_SIZES = [1200, 1600, 2000, 2400, 2800]


# ============================================================
# send with raised chunk cap
# ============================================================

def cmd_send(args):
    if not 100 <= args.chunk_size <= LAB_MAX_CHUNK:
        raise SystemExit(
            f"--chunk-size must be between 100 and {LAB_MAX_CHUNK} bytes"
        )

    qrtx.MAX_CHUNK_SIZE = LAB_MAX_CHUNK  # lift the safety cap
    qrtx.send_files(
        args.files, args.fps, args.chunk_size,
        args.monitor, args.fullscreen, args.codes,
    )


# ============================================================
# Calibration
#
# Frame: QLC:SIZE:SEQ:CRC32:DATA45 with a per-size sequence number,
# so the receiver can compute decode rate as
#   distinct seqs seen / (max seq - min seq + 1)
# even when the sender cycles through the sizes repeatedly.
# ============================================================

def cal_payload(size: int, seq: int) -> str:
    data = qrtx.whiten(os.urandom(size), seq)
    crc = f"{zlib.crc32(data) & 0xffffffff:08X}"
    return ":".join(
        [CAL_MAGIC, str(size), str(seq), crc, qrtx.b45encode(data)]
    )


def parse_cal(text: str) -> tuple[int, int] | None:
    try:
        if not text.startswith(CAL_MAGIC + ":"):
            return None
        parts = text.split(":", 4)
        if len(parts) != 5:
            return None
        _, size_s, seq_s, crc, data45 = parts

        size = int(size_s)
        seq = int(seq_s)
        data = qrtx.b45decode(data45)

        if len(data) != size or size <= 0 or seq < 0:
            return None
        if f"{zlib.crc32(data) & 0xffffffff:08X}" != crc.upper():
            return None

        return size, seq
    except Exception:
        return None


def cmd_caltx(args):
    import cv2

    sizes = args.sizes or CAL_SIZES
    dwell_frames = max(1, int(args.seconds * args.fps))
    delay = 1.0 / args.fps
    seqs = {s: 0 for s in sizes}

    print()
    print("QRTX calibration sender")
    print("=" * 50)
    print(f"Cycling chunk sizes {sizes}, {args.seconds:g}s each.")
    print("Run 'qrtx_lab.py calrx' on the receiver, let it watch a few")
    print("full cycles, then press Q there to see the report.")
    print("Press Q or Esc here to stop.")

    cv2.namedWindow("QRTX Calibration", cv2.WINDOW_NORMAL)
    first = qrtx.payload_to_qr(cal_payload(sizes[0], 0))
    shown = qrtx.add_sender_status(first, "warming up", "")
    qrtx.place_window(
        "QRTX Calibration",
        (shown.shape[1], shown.shape[0]),
        args.monitor,
        args.fullscreen,
    )

    try:
        while True:
            for size in sizes:
                for _ in range(dwell_frames):
                    t0 = time.time()

                    seq = seqs[size]
                    seqs[size] += 1
                    img = qrtx.payload_to_qr(cal_payload(size, seq))
                    shown = qrtx.add_sender_status(
                        img,
                        f"calibration: chunk {size} B  seq {seq}",
                        f"{args.fps:g} FPS   sizes {sizes}",
                    )
                    cv2.imshow("QRTX Calibration", shown)

                    elapsed = int((time.time() - t0) * 1000)
                    wait = max(1, int(delay * 1000) - elapsed)
                    if (cv2.waitKey(wait) & 0xFF) in (27, ord("q"), ord("Q")):
                        return
    finally:
        cv2.destroyAllWindows()


def cmd_calrx(args):
    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    got = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print()
    print("QRTX calibration receiver")
    print("=" * 50)
    print(f"Camera resolution: {got[0]}x{got[1]}")
    print("Point the camera at the calibration sender.")
    print("Press Q or Esc to finish and print the report.")

    detector = qrtx.make_detector()
    reader = qrtx.LatestFrameReader(cap)
    stats: dict[int, dict] = {}
    seq_no = 0

    cv2.namedWindow("QRTX Calibration RX", cv2.WINDOW_NORMAL)
    qrtx.place_window("QRTX Calibration RX", (1000, 700), args.monitor)

    try:
        while True:
            seq_no, frame = reader.read(seq_no)
            if frame is None:
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                    break
                continue

            found, texts, _, _ = detector.detectAndDecodeMulti(frame)
            for text in (texts if found else ()):
                parsed = parse_cal(text) if text else None
                if parsed is None:
                    continue
                size, seq = parsed
                st = stats.setdefault(
                    size, {"seen": set(), "lo": seq, "hi": seq},
                )
                st["seen"].add(seq)
                st["lo"] = min(st["lo"], seq)
                st["hi"] = max(st["hi"], seq)

            lines = ["QRTX calibration: decode rate per chunk size"]
            for size in sorted(stats):
                st = stats[size]
                shown_ct = st["hi"] - st["lo"] + 1
                rate = len(st["seen"]) / shown_ct
                lines.append(
                    f"{size:>5} B: {100 * rate:5.1f}%  "
                    f"({len(st['seen'])}/{shown_ct})"
                )

            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], 34 + 30 * len(lines)),
                (255, 255, 255), -1,
            )
            for i, line in enumerate(lines):
                cv2.putText(
                    frame, line, (12, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2,
                    cv2.LINE_AA,
                )

            cv2.imshow("QRTX Calibration RX", frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                break
    finally:
        reader.stop()
        cap.release()
        cv2.destroyAllWindows()

    print()
    print("Decode rate per chunk size:")
    best = None
    for size in sorted(stats):
        st = stats[size]
        shown_ct = st["hi"] - st["lo"] + 1
        rate = len(st["seen"]) / shown_ct
        print(f"  {size:>5} B: {100 * rate:5.1f}%  "
              f"({len(st['seen'])}/{shown_ct} frames)")
        if rate >= 0.8:
            best = max(best or 0, size)

    print()
    if best is None:
        print("No size reached 80% decode rate. Move the camera closer,")
        print("enlarge the sender window, or improve focus, then retry.")
    else:
        tool = "qrtx.py" if best <= 1200 else "qrtx_lab.py"
        print(f"Recommended: --chunk-size {best} (via {tool} send)")


# ============================================================
# Color multiplexing: one protocol frame per RGB channel
# ============================================================

def _gray_qr(payload: str):
    import cv2

    return cv2.cvtColor(qrtx.payload_to_qr(payload), cv2.COLOR_BGR2GRAY)


def _pad_center(gray, side: int):
    import numpy as np

    canvas = np.full((side, side), 255, dtype=gray.dtype)
    y = (side - gray.shape[0]) // 2
    x = (side - gray.shape[1]) // 2
    canvas[y:y + gray.shape[0], x:x + gray.shape[1]] = gray
    return canvas


def color_frame(meta: dict, chunks: list[bytes], seeds: list[int]):
    """Three droplets in one image: B, G and R each carry a QR code."""
    import cv2

    grays = [
        _gray_qr(qrtx.data_payload(meta, chunks, s))
        for s in seeds
    ]
    side = max(g.shape[0] for g in grays)
    return cv2.merge([_pad_center(g, side) for g in grays])


def cmd_csend(args):
    import cv2

    paths = qrtx.expand_patterns(args.files)
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
    if not 100 <= args.chunk_size <= LAB_MAX_CHUNK:
        raise SystemExit(
            f"--chunk-size must be between 100 and {LAB_MAX_CHUNK} bytes"
        )

    if len(paths) == 1:
        name, raw, kind = paths[0].name, paths[0].read_bytes(), qrtx.KIND_FILE
    else:
        (name, raw), kind = qrtx.bundle_files(paths), qrtx.KIND_BUNDLE

    meta, chunks = qrtx.build_session(name, raw, kind, args.chunk_size)
    n = meta["n"]
    header_img = qrtx.payload_to_qr(qrtx.header_payload(meta))
    delay = 1.0 / args.fps

    print()
    print("QRTX color sender (experimental, 3 frames per image)")
    print("=" * 50)
    print(f"File:       {name}")
    print(f"Compressed: {meta['zlen']:,} bytes  Chunks: {n}")
    print(f"Session:    {meta['sid']}")
    print("Press Q or Esc in the window to stop.")

    cv2.namedWindow("QRTX Color Sender", cv2.WINDOW_NORMAL)

    frame_no = 0
    data_sent = 0
    placed = False

    try:
        while True:
            t0 = time.time()

            if frame_no % qrtx.HEADER_EVERY == 0:
                # Header goes out grayscale (identical in all channels)
                # so it decodes like a normal QR code.
                img = header_img
                label = "header"
            else:
                seeds = []
                for _ in range(3):
                    seeds.append(
                        data_sent if data_sent < n
                        else random.randrange(n, 1 << 31)
                    )
                    data_sent += 1
                img = color_frame(meta, chunks, seeds)
                label = f"droplets {data_sent} (3 per image)"

            shown = qrtx.add_sender_status(
                img,
                f"{label}   {n} chunks",
                f"session {meta['sid']}   COLOR x3   {args.fps:g} FPS",
            )

            if not placed:
                qrtx.place_window(
                    "QRTX Color Sender",
                    (shown.shape[1], shown.shape[0]),
                    args.monitor,
                    args.fullscreen,
                )
                placed = True

            cv2.imshow("QRTX Color Sender", shown)

            elapsed = int((time.time() - t0) * 1000)
            wait = max(1, int(delay * 1000) - elapsed)
            if (cv2.waitKey(wait) & 0xFF) in (27, ord("q"), ord("Q")):
                return

            frame_no += 1
    finally:
        cv2.destroyAllWindows()


def cmd_creceive(args):
    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print()
    print("QRTX color receiver (experimental)")
    print("=" * 50)
    print(f"Camera resolution: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Press Q or Esc to quit.")

    detector = qrtx.make_detector()
    reader = qrtx.LatestFrameReader(cap)

    lock = None
    header = None
    decoder = None
    seq_no = 0

    cv2.namedWindow("QRTX Color Receiver", cv2.WINDOW_NORMAL)
    qrtx.place_window("QRTX Color Receiver", (1000, 700), args.monitor)

    def ingest(text: str) -> bool:
        """Feed one decoded string; returns True when transfer completes."""
        nonlocal lock, header, decoder

        parsed = qrtx.parse_payload(text) if text else None
        if parsed is None:
            return False
        kind, obj = parsed

        if lock is None:
            lock = (obj["sid"], obj["n"])
            decoder = qrtx.FountainDecoder(obj["n"])
            print(f"Locked session {lock[0]} ({lock[1]} chunks)")
        if (obj["sid"], obj["n"]) != lock:
            return False

        if kind == "header":
            if header is None:
                header = obj
                print(f"Header: {header['name']!r}")
        else:
            before = len(decoder.recovered)
            decoder.add(obj["seed"], obj["droplet"])
            if len(decoder.recovered) > before:
                print(
                    f"\rChunks {len(decoder.recovered)}/{decoder.n}",
                    end="",
                    flush=True,
                )

        return decoder.complete and header is not None

    try:
        while True:
            seq_no, frame = reader.read(seq_no)
            if frame is None:
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                    return
                continue

            done = False
            # Each channel carries its own QR; header frames are
            # grayscale so any single channel decodes them too.
            for channel in cv2.split(frame):
                found, texts, _, _ = detector.detectAndDecodeMulti(channel)
                for text in (texts if found else ()):
                    done = ingest(text) or done

            if done:
                print()
                print("All chunks recovered. Verifying...")
                out = qrtx.finalize_received(
                    header, decoder, args.out, args.overwrite,
                )
                print(f"Saved: {out.resolve()}")
                return

            qrtx.draw_receiver_status(
                frame,
                len(decoder.recovered) if decoder else 0,
                decoder.n if decoder else None,
                decoder.droplets_used if decoder else 0,
                header is not None,
                lock[0] if lock else None,
            )
            cv2.imshow("QRTX Color Receiver", frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                return
    finally:
        reader.stop()
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================

def _add_window_args(sp, fullscreen: bool = True):
    sp.add_argument("--monitor", type=int, default=None)
    if fullscreen:
        sp.add_argument("--fullscreen", action="store_true")


def _add_camera_args(sp):
    sp.add_argument("--camera", type=int, default=0)
    sp.add_argument("--width", type=int, default=1920)
    sp.add_argument("--height", type=int, default=1080)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qrtx_lab",
        description="Experimental qrtx extensions (see module docstring).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser(
        "send", help=f"qrtx send with chunk sizes up to {LAB_MAX_CHUNK} B",
    )
    ps.add_argument("files", type=Path, nargs="+")
    ps.add_argument("--fps", type=float, default=qrtx.DEFAULT_FPS)
    ps.add_argument("--chunk-size", type=int, default=1600)
    ps.add_argument("--codes", type=int, default=qrtx.DEFAULT_CODES)
    _add_window_args(ps)

    pt = sub.add_parser(
        "caltx", help="Show cycling calibration codes on this screen",
    )
    pt.add_argument("--fps", type=float, default=qrtx.DEFAULT_FPS)
    pt.add_argument("--seconds", type=float, default=10.0,
                    help="Dwell time per chunk size")
    pt.add_argument("--sizes", type=int, nargs="*", default=None)
    _add_window_args(pt)

    pc = sub.add_parser(
        "calrx", help="Watch calibration codes and report decode rates",
    )
    _add_camera_args(pc)
    pc.add_argument("--monitor", type=int, default=None)

    pcs = sub.add_parser(
        "csend", help="Color-multiplexed send (3 frames per image)",
    )
    pcs.add_argument("files", type=Path, nargs="+")
    pcs.add_argument("--fps", type=float, default=qrtx.DEFAULT_FPS)
    pcs.add_argument("--chunk-size", type=int,
                     default=qrtx.DEFAULT_CHUNK_SIZE)
    _add_window_args(pcs)

    pcr = sub.add_parser("creceive", help="Color-multiplexed receive")
    _add_camera_args(pcr)
    pcr.add_argument("--out", type=Path, default=Path("received"))
    pcr.add_argument("--overwrite", action="store_true")
    pcr.add_argument("--monitor", type=int, default=None)

    return p


def main():
    qrtx.require_runtime_deps()
    args = build_parser().parse_args()

    {
        "send": cmd_send,
        "caltx": cmd_caltx,
        "calrx": cmd_calrx,
        "csend": cmd_csend,
        "creceive": cmd_creceive,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
