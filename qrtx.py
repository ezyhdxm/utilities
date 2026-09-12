#!/usr/bin/env python3
"""Offline file transfer using animated QR codes.

Protocol v2 highlights:

- LT fountain coding: the sender streams an endless supply of XOR-combined
  chunks ("droplets"), so the receiver only needs *enough* frames, not every
  specific frame. The first pass is systematic (plain chunks in order), so
  small files complete in one clean pass.
- QR alphanumeric mode + base45 payload encoding (~3% expansion instead of
  base85's 25% in byte mode).
- Compact frames: session metadata (filename, sizes, SHA-256) travels in a
  periodic header frame instead of being repeated in every data frame.
- Error correction level L: frame-level CRC + fountain retransmission make
  in-frame redundancy unnecessary on a screen-to-camera link.
- Best-of zlib/lzma compression, chosen per file.
- Receiver uses QRCodeDetectorAruco when available and a threaded frame
  grabber so decoding never falls behind the camera.
- Multiple files (`qrtx send *.py`) are packed into one in-memory tar
  bundle and sent as a single session; the receiver auto-extracts it,
  preserving paths relative to the files' common parent folder.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import hashlib
import io
import lzma
import math
import os
import random
import sys
import tarfile
import threading
import time
import zlib
from pathlib import Path


MAGIC_HEADER = "Q2H"
MAGIC_DATA = "Q2D"

# Bytes of (compressed) payload carried by each QR data frame.
DEFAULT_CHUNK_SIZE = 900

DEFAULT_FPS = 6.0

# A header frame is inserted every N displayed frames.
HEADER_EVERY = 15


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
    return hashlib.sha256(data).hexdigest().upper()


# ============================================================
# Base45 (RFC 9285)
#
# The base45 alphabet is exactly the QR alphanumeric charset, which packs
# at 5.5 bits/char instead of byte mode's 8 bits/char. Two bytes become
# three chars (16.5 bits for 16 bits of data): ~3% overhead.
# ============================================================

B45_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
_B45_REVERSE = {c: i for i, c in enumerate(B45_ALPHABET)}


def b45encode(data: bytes) -> str:
    out: list[str] = []
    a = B45_ALPHABET

    for i in range(0, len(data) - 1, 2):
        v = (data[i] << 8) | data[i + 1]
        v, c = divmod(v, 45)
        e, d = divmod(v, 45)
        out.append(a[c])
        out.append(a[d])
        out.append(a[e])

    if len(data) % 2:
        e, c = divmod(data[-1], 45)
        out.append(a[c])
        out.append(a[e])

    return "".join(out)


def b45decode(text: str) -> bytes:
    if len(text) % 3 == 1:
        raise ValueError("invalid base45 length")

    out = bytearray()
    rev = _B45_REVERSE

    for i in range(0, len(text) - 2, 3):
        v = rev[text[i]] + rev[text[i + 1]] * 45 + rev[text[i + 2]] * 2025
        if v > 0xFFFF:
            raise ValueError("invalid base45 triple")
        out.append(v >> 8)
        out.append(v & 0xFF)

    if len(text) % 3 == 2:
        v = rev[text[-2]] + rev[text[-1]] * 45
        if v > 0xFF:
            raise ValueError("invalid base45 pair")
        out.append(v)

    return bytes(out)


def encode_filename(name: str) -> str:
    return b45encode(name.encode("utf-8"))


def decode_filename(encoded: str) -> str:
    return b45decode(encoded).decode("utf-8", errors="replace")


# ============================================================
# Compression
# ============================================================

def compress_best(raw: bytes) -> tuple[str, bytes]:
    """Return (algo, data) using whichever of zlib/lzma is smaller."""
    z = zlib.compress(raw, 9)
    x = lzma.compress(raw, preset=6)
    return ("X", x) if len(x) < len(z) else ("Z", z)


def decompress(algo: str, data: bytes) -> bytes:
    if algo == "X":
        return lzma.decompress(data)
    if algo == "Z":
        return zlib.decompress(data)
    raise ValueError(f"unknown compression algo {algo!r}")


# ============================================================
# LT fountain code
# ============================================================

_CDF_CACHE: dict[int, list[float]] = {}


def _degree_cdf(n: int, c: float = 0.05, delta: float = 0.05) -> list[float]:
    """Cumulative robust-soliton degree distribution for n chunks."""
    cached = _CDF_CACHE.get(n)
    if cached is not None:
        return cached

    if n == 1:
        cdf = [1.0]
        _CDF_CACHE[n] = cdf
        return cdf

    r = c * math.log(n / delta) * math.sqrt(n)
    spike = min(n, max(1, int(round(n / r)))) if r > 0 else n

    probs = [0.0] * (n + 1)
    probs[1] = 1.0 / n
    for d in range(2, n + 1):
        probs[d] = 1.0 / (d * (d - 1))

    if r > 0:
        for d in range(1, spike):
            probs[d] += r / (d * n)
        probs[spike] += max(0.0, r * math.log(r / delta) / n)

    total = sum(probs)
    acc = 0.0
    cdf = []
    for d in range(1, n + 1):
        acc += probs[d] / total
        cdf.append(acc)
    cdf[-1] = 1.0

    _CDF_CACHE[n] = cdf
    return cdf


def droplet_indices(seed: int, n: int) -> list[int]:
    """Chunk indices XOR-combined in the droplet for `seed`.

    Seeds 0..n-1 are systematic (droplet == that single chunk); larger
    seeds derive a pseudo-random combination. Sender and receiver must
    run the same code for these to agree.
    """
    if seed < n:
        return [seed]

    rng = random.Random(seed)
    cdf = _degree_cdf(n)
    degree = min(n, bisect.bisect_left(cdf, rng.random()) + 1)
    return rng.sample(range(n), degree)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    return (
        int.from_bytes(a, "big") ^ int.from_bytes(b, "big")
    ).to_bytes(len(a), "big")


def whiten(data: bytes, seed: int) -> bytes:
    """XOR with a seed-derived pseudorandom keystream (symmetric).

    The last chunk of a session is zero-padded, and zero-heavy QR
    payloads crash python-qrcode's Reed-Solomon encoder with
    "ValueError: glog(0)" (lincolnloop/python-qrcode#330, still present
    in 8.2). Whitening makes every wire payload look pseudorandom, so
    the bug can never trigger regardless of file content. Sender and
    receiver derive the same keystream from the droplet seed.
    """
    rng = random.Random(seed ^ 0x9E3779B9)
    return xor_bytes(data, rng.randbytes(len(data)))


def make_droplet(chunks: list[bytes], seed: int) -> bytes:
    idxs = droplet_indices(seed, len(chunks))
    out = chunks[idxs[0]]
    for i in idxs[1:]:
        out = xor_bytes(out, chunks[i])
    return out


class FountainDecoder:
    """Peeling decoder for LT droplets."""

    def __init__(self, n: int):
        self.n = n
        self.recovered: dict[int, bytes] = {}
        self.droplets_used = 0
        self._chunk_len: int | None = None
        self._pending: dict[int, tuple[set[int], bytes]] = {}
        self._by_idx: dict[int, set[int]] = {}
        self._next_id = 0
        self._seen_seeds: set[int] = set()

    @property
    def complete(self) -> bool:
        return len(self.recovered) == self.n

    def add(self, seed: int, payload: bytes) -> None:
        if self.complete or seed in self._seen_seeds:
            return

        if self._chunk_len is None:
            self._chunk_len = len(payload)
        elif len(payload) != self._chunk_len:
            return

        self._seen_seeds.add(seed)
        self.droplets_used += 1

        idxs = set(droplet_indices(seed, self.n))
        for i in list(idxs):
            if i in self.recovered:
                payload = xor_bytes(payload, self.recovered[i])
                idxs.discard(i)

        if not idxs:
            return

        if len(idxs) == 1:
            self._recover(idxs.pop(), payload)
            return

        did = self._next_id
        self._next_id += 1
        self._pending[did] = (idxs, payload)
        for i in idxs:
            self._by_idx.setdefault(i, set()).add(did)

    def _recover(self, idx: int, data: bytes) -> None:
        stack = [(idx, data)]

        while stack:
            i, d = stack.pop()
            if i in self.recovered:
                continue
            self.recovered[i] = d

            for did in list(self._by_idx.get(i, ())):
                idxs, payload = self._pending[did]
                payload = xor_bytes(payload, d)
                idxs.discard(i)
                self._by_idx[i].discard(did)

                if len(idxs) == 1:
                    j = next(iter(idxs))
                    del self._pending[did]
                    self._by_idx.get(j, set()).discard(did)
                    stack.append((j, payload))
                else:
                    self._pending[did] = (idxs, payload)

    def assemble(self) -> bytes:
        return b"".join(self.recovered[i] for i in range(self.n))


# ============================================================
# Protocol / frames
#
# Frames are strings restricted to the QR alphanumeric charset so the
# whole frame packs at 5.5 bits/char. ':' also appears inside base45
# output, so every frame keeps its single free-form field last and is
# parsed with a bounded split.
#
#   header: Q2H:SID:N:CHUNK:ALGO:KIND:ZLEN:OLEN:SHA:FNAME45
#   data:   Q2D:SID:N:SEED:CRC32:DATA45
#
# DATA45 carries the droplet whitened with a seed-derived keystream
# (see whiten()); CRC32 covers the whitened bytes as transmitted.
#
# KIND is F for a single file, B for a tar bundle of several files
# (the receiver extracts bundles automatically).
# ============================================================

KIND_FILE = "F"
KIND_BUNDLE = "B"


def _common_parent(paths: list[Path]) -> Path | None:
    """Deepest common parent folder of the given files, if any."""
    try:
        common = Path(os.path.commonpath([p.resolve() for p in paths]))
    except ValueError:
        return None

    if common.is_file():  # the same single file given more than once
        common = common.parent

    return common


def bundle_name(paths: list[Path]) -> str:
    """Name a bundle after the files' deepest common parent folder.

    Sending project/*.py yields "project.tar", so the receiver extracts
    into received/project/ instead of an anonymous bundle_N_files/.
    Falls back to a generic name when there is no meaningful common
    folder (e.g. files spread across filesystem roots).
    """
    common = _common_parent(paths)
    stem = common.name if common is not None else ""

    if not stem:
        stem = f"bundle_{len(paths)}_files"

    return f"{stem}.tar"


def bundle_files(paths: list[Path]) -> tuple[str, bytes]:
    """Pack several files into an in-memory tar; returns (name, bytes).

    Each file is stored under its path relative to the files' common
    parent folder, so 'send sub/a/*.py sub/b/*.py' arrives as
    received/sub/a/... and received/sub/b/..., mirroring the sender's
    layout. Without a common folder, files are stored flat by basename.
    """
    buf = io.BytesIO()
    used: set[str] = set()
    common = _common_parent(paths)

    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in paths:
            if common is not None:
                arcname = p.resolve().relative_to(common).as_posix()
                if arcname in used:  # exact same file listed twice
                    continue
            else:
                arcname = p.name
                i = 1
                while arcname in used:
                    arcname = f"{p.stem}_{i}{p.suffix}"
                    i += 1

            used.add(arcname)
            tar.add(p, arcname=arcname, recursive=False)

    return bundle_name(paths), buf.getvalue()


def build_session(
    name: str,
    raw: bytes,
    kind: str,
    chunk_size: int,
) -> tuple[dict, list[bytes]]:
    sha = file_sha256(raw)
    algo, comp = compress_best(raw)

    zlen = len(comp)
    n = max(1, math.ceil(zlen / chunk_size))
    padded = comp.ljust(n * chunk_size, b"\x00")
    chunks = [
        padded[i * chunk_size:(i + 1) * chunk_size]
        for i in range(n)
    ]

    meta = {
        "sid": f"{sha[:4]}{random.randrange(16 ** 4):04X}",
        "n": n,
        "chunk": chunk_size,
        "algo": algo,
        "kind": kind,
        "zlen": zlen,
        "olen": len(raw),
        "sha": sha,
        "name": name,
    }
    return meta, chunks


def header_payload(meta: dict) -> str:
    return ":".join([
        MAGIC_HEADER,
        meta["sid"],
        str(meta["n"]),
        str(meta["chunk"]),
        meta["algo"],
        meta["kind"],
        str(meta["zlen"]),
        str(meta["olen"]),
        meta["sha"],
        encode_filename(meta["name"]),
    ])


def data_payload(meta: dict, chunks: list[bytes], seed: int) -> str:
    droplet = whiten(make_droplet(chunks, seed), seed)
    crc = f"{zlib.crc32(droplet) & 0xffffffff:08X}"
    return ":".join([
        MAGIC_DATA,
        meta["sid"],
        str(meta["n"]),
        str(seed),
        crc,
        b45encode(droplet),
    ])


def parse_payload(text: str) -> tuple[str, dict] | None:
    try:
        if text.startswith(MAGIC_DATA + ":"):
            parts = text.split(":", 5)
            if len(parts) != 6:
                return None
            _, sid, n_s, seed_s, crc, data45 = parts

            n = int(n_s)
            seed = int(seed_s)
            if n <= 0 or seed < 0:
                return None

            droplet = b45decode(data45)
            if f"{zlib.crc32(droplet) & 0xffffffff:08X}" != crc.upper():
                return None

            return "data", {
                "sid": sid,
                "n": n,
                "seed": seed,
                "droplet": whiten(droplet, seed),
            }

        if text.startswith(MAGIC_HEADER + ":"):
            parts = text.split(":", 9)
            if len(parts) != 10:
                return None
            (
                _, sid, n_s, chunk_s, algo, kind,
                zlen_s, olen_s, sha, fn45,
            ) = parts

            n = int(n_s)
            chunk = int(chunk_s)
            zlen = int(zlen_s)
            olen = int(olen_s)

            if n <= 0 or chunk <= 0 or zlen < 0 or olen < 0:
                return None
            if algo not in ("Z", "X") or len(sha) != 64:
                return None
            if kind not in (KIND_FILE, KIND_BUNDLE):
                return None

            return "header", {
                "sid": sid,
                "n": n,
                "chunk": chunk,
                "algo": algo,
                "kind": kind,
                "zlen": zlen,
                "olen": olen,
                "sha": sha,
                "name": decode_filename(fn45),
            }

        return None

    except Exception:
        return None


# ============================================================
# QR generation
# ============================================================

def payload_to_qr(payload: str):
    import cv2
    import numpy as np
    import qrcode

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    # Render at 1 px/module straight from the matrix (no PIL round trip),
    # then upscale with nearest-neighbour: much faster per frame.
    matrix = np.array(qr.get_matrix(), dtype=np.uint8)
    gray = (1 - matrix) * np.uint8(255)

    px = max(3, min(10, 940 // gray.shape[0]))
    gray = cv2.resize(
        gray,
        None,
        fx=px,
        fy=px,
        interpolation=cv2.INTER_NEAREST,
    )
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

def expand_patterns(paths: list[Path]) -> list[Path]:
    """Expand wildcard patterns that the shell didn't.

    Unix shells expand globs before the program runs, but Windows cmd
    (and PowerShell) pass patterns like folder\\*.py through literally.
    Non-file matches (directories) are skipped.
    """
    out: list[Path] = []

    for p in paths:
        s = str(p)
        if any(ch in s for ch in "*?[") and not p.exists():
            matches = [Path(m) for m in sorted(glob.glob(s))]
            files = [m for m in matches if m.is_file()]
            if not files:
                raise SystemExit(f"No files match pattern: {s}")
            out.extend(files)
        else:
            out.append(p)

    return out


def enumerate_monitors() -> list[tuple[int, int, int, int]]:
    """Best-effort list of monitors as (x, y, width, height), primary first.

    Returns an empty list when the layout can't be determined, in which
    case callers fall back to OpenCV's default window placement.
    """
    monitors: list[tuple[int, int, int, int]] = []

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            proc = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(wintypes.RECT),
                ctypes.c_void_p,
            )

            def callback(_hmon, _hdc, rect, _data):
                r = rect.contents
                monitors.append(
                    (r.left, r.top, r.right - r.left, r.bottom - r.top)
                )
                return 1

            ctypes.windll.user32.EnumDisplayMonitors(
                None, None, proc(callback), 0
            )
        except Exception:
            monitors = []

    if not monitors:
        try:
            import screeninfo

            monitors = [
                (m.x, m.y, m.width, m.height)
                for m in screeninfo.get_monitors()
            ]
        except Exception:
            monitors = []

    # Primary monitor sits at the origin; show it first so --monitor 1
    # means "the main screen" on every platform.
    monitors.sort(key=lambda m: (m[0] != 0 or m[1] != 0, m[0], m[1]))
    return monitors


def place_window(
    name: str,
    size: tuple[int, int],
    monitor: int | None = None,
    fullscreen: bool = False,
):
    """Size and position a window so it never straddles two monitors."""
    import cv2

    mons = enumerate_monitors()

    if monitor is not None:
        if not mons:
            raise SystemExit(
                "--monitor: could not detect monitors on this system "
                "(try: python -m pip install screeninfo)"
            )
        if not 1 <= monitor <= len(mons):
            raise SystemExit(
                f"--monitor must be between 1 and {len(mons)}; "
                "run 'qrtx monitors' to list them"
            )
        x, y, mw, mh = mons[monitor - 1]
    elif mons:
        x, y, mw, mh = mons[0]
    else:
        cv2.resizeWindow(name, *size)
        return

    if fullscreen:
        cv2.moveWindow(name, x, y)
        cv2.setWindowProperty(
            name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        return

    # Fill ~90% of the chosen monitor while keeping the content's aspect
    # ratio, so the QR is rendered as large as that screen allows.
    aspect = size[0] / size[1]
    h = int(mh * 0.9)
    w = int(h * aspect)
    if w > mw * 0.9:
        w = int(mw * 0.9)
        h = int(w / aspect)

    cv2.resizeWindow(name, w, h)
    cv2.moveWindow(name, x + (mw - w) // 2, y + (mh - h) // 2)


def print_monitors():
    mons = enumerate_monitors()

    if not mons:
        print(
            "Could not detect monitors on this system.\n"
            "Install 'screeninfo' for detection, or just drag the window."
        )
        return

    print(f"{len(mons)} monitor(s) detected:")
    for i, (x, y, w, h) in enumerate(mons, 1):
        tag = "  (primary)" if x == 0 and y == 0 else ""
        print(f"  --monitor {i}   {w}x{h} at ({x}, {y}){tag}")


def send_files(
    paths: list[Path],
    fps: float,
    chunk_size: int,
    monitor: int | None = None,
    fullscreen: bool = False,
):
    import cv2

    paths = expand_patterns(paths)

    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")

    if fps <= 0:
        raise SystemExit("--fps must be > 0")

    if not 100 <= chunk_size <= 1200:
        raise SystemExit("--chunk-size must be between 100 and 1200 bytes")

    if len(paths) == 1:
        name = paths[0].name
        raw = paths[0].read_bytes()
        kind = KIND_FILE
    else:
        name, raw = bundle_files(paths)
        kind = KIND_BUNDLE

    meta, chunks = build_session(name, raw, kind, chunk_size)
    n = meta["n"]

    header_img = payload_to_qr(header_payload(meta))
    delay = 1.0 / fps

    print()
    print("QRTX sender (v2, fountain-coded)")
    print("=" * 50)
    if kind == KIND_BUNDLE:
        print(f"Bundle:     {len(paths)} files -> {name}")
        for p in paths:
            print(f"            - {p.name}")
    else:
        print(f"File:       {name}")
    print(f"Original:   {meta['olen']:,} bytes")
    print(f"Compressed: {meta['zlen']:,} bytes ({meta['algo']})")
    print(f"Chunks:     {n}")
    print(f"Session:    {meta['sid']}")
    print()
    print("Press Q or Esc in the QR window to stop.")

    cv2.namedWindow("QRTX Sender", cv2.WINDOW_NORMAL)
    place_window("QRTX Sender", (900, 980), monitor, fullscreen)

    frame_no = 0
    data_sent = 0

    try:
        while True:
            t0 = time.time()

            if frame_no % HEADER_EVERY == 0:
                img = header_img
                label = "header"
            else:
                # Systematic first pass, then endless random droplets.
                if data_sent < n:
                    seed = data_sent
                else:
                    seed = random.randrange(n, 1 << 31)
                data_sent += 1

                img = payload_to_qr(data_payload(meta, chunks, seed))
                cycle = data_sent // n if n else 0
                label = f"droplet {data_sent} (seed {seed}, pass {cycle + 1})"

            shown = add_sender_status(
                img,
                f"{label}   {n} chunks",
                f"session {meta['sid']}   {fps:g} FPS   "
                f"compressed {meta['zlen']:,} B",
            )

            cv2.imshow("QRTX Sender", shown)

            elapsed_ms = int((time.time() - t0) * 1000)
            wait_ms = max(1, int(delay * 1000) - elapsed_ms)
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                return

            frame_no += 1

    finally:
        cv2.destroyAllWindows()


# ============================================================
# Receiver output helpers
# ============================================================

def choose_output_path(
    out_dir: Path,
    filename: str,
    overwrite: bool = False,
) -> Path:
    safe_name = Path(filename).name or "received.bin"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = out_dir / safe_name
    if overwrite or not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix

    for i in range(1, 10000):
        alt = out_dir / f"{stem}_received_{i}{suffix}"
        if not alt.exists():
            return alt

    raise RuntimeError("Could not choose a free output filename")


def choose_output_dir(
    out_dir: Path,
    name: str,
    overwrite: bool = False,
) -> Path:
    safe_name = Path(name).name or "bundle"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = out_dir / safe_name
    if overwrite or not candidate.exists():
        return candidate

    for i in range(1, 10000):
        alt = out_dir / f"{safe_name}_{i}"
        if not alt.exists():
            return alt

    raise RuntimeError("Could not choose a free output directory")


def extract_bundle(
    raw: bytes,
    out_dir: Path,
    name: str,
    overwrite: bool = False,
) -> Path:
    target = choose_output_dir(out_dir, Path(name).stem, overwrite)
    target.mkdir(exist_ok=overwrite)

    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        tar.extractall(target, filter="data")

    return target


def finalize_received(
    header: dict,
    decoder: FountainDecoder,
    out_dir: Path,
    overwrite: bool = False,
) -> Path:
    compressed = decoder.assemble()[: header["zlen"]]

    if len(compressed) != header["zlen"]:
        raise ValueError(
            "Compressed size mismatch: "
            f"expected {header['zlen']}, got {len(compressed)}"
        )

    raw = decompress(header["algo"], compressed)

    if len(raw) != header["olen"]:
        raise ValueError(
            "Original size mismatch: "
            f"expected {header['olen']}, got {len(raw)}"
        )

    if file_sha256(raw) != header["sha"].upper():
        raise ValueError("SHA-256 verification failed")

    if header["kind"] == KIND_BUNDLE:
        return extract_bundle(raw, out_dir, header["name"], overwrite)

    out_path = choose_output_path(out_dir, header["name"], overwrite)
    temp_path = out_path.with_name(out_path.name + ".part")

    temp_path.write_bytes(raw)
    temp_path.replace(out_path)

    return out_path


# ============================================================
# Receiver UI
# ============================================================

def draw_receiver_status(
    frame,
    recovered: int,
    total: int | None,
    droplets: int,
    have_header: bool,
    session: str | None,
):
    import cv2

    if total:
        pct = 100.0 * recovered / total
        line1 = f"QRTX: {recovered}/{total} chunks ({pct:.1f}%)"
    else:
        line1 = "QRTX: waiting for first frame..."

    hdr = "yes" if have_header else "waiting"
    line2 = (
        f"session: {session or '-'}   droplets: {droplets}   "
        f"header: {hdr}   Q/Esc to quit"
    )

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

class LatestFrameReader:
    """Grabs camera frames on a thread, keeping only the newest one.

    Decoding a frame can take longer than the camera's frame interval;
    without this, stale frames pile up in the driver buffer and the
    receiver decodes progressively older images.
    """

    def __init__(self, cap):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq: int, timeout: float = 0.5):
        """Return (seq, frame) newer than last_seq, or (last_seq, None)."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq > last_seq or self._stopped,
                timeout=timeout,
            )
            if self._seq > last_seq and self._frame is not None:
                return self._seq, self._frame
            return last_seq, None

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()


def make_detector():
    import cv2

    # QRCodeDetectorAruco (OpenCV >= 4.8) is faster and more robust.
    if hasattr(cv2, "QRCodeDetectorAruco"):
        return cv2.QRCodeDetectorAruco()
    return cv2.QRCodeDetector()


def receive_file(
    camera: int,
    out_dir: Path,
    width: int,
    height: int,
    overwrite: bool = False,
    monitor: int | None = None,
):
    import cv2

    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = make_detector()
    reader = LatestFrameReader(cap)

    lock: tuple[str, int] | None = None
    header: dict | None = None
    decoder: FountainDecoder | None = None
    last_new = 0.0
    seq = 0

    print()
    print("QRTX receiver (v2, fountain-coded)")
    print("=" * 50)
    print("Point the camera at the sender screen.")
    print("Receiver will lock onto the first QRTX session.")
    print("Press Q or Esc to quit.")

    cv2.namedWindow("QRTX Receiver", cv2.WINDOW_NORMAL)
    place_window("QRTX Receiver", (1000, 700), monitor)

    try:
        while True:
            seq, frame = reader.read(seq)

            if frame is None:
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                    return
                continue

            text, points, _ = detector.detectAndDecode(frame)
            parsed = parse_payload(text) if text else None

            if parsed is not None:
                kind, obj = parsed

                if lock is None:
                    lock = (obj["sid"], obj["n"])
                    decoder = FountainDecoder(obj["n"])
                    print(f"Locked session {lock[0]} ({lock[1]} chunks)")

                assert decoder is not None

                if (obj["sid"], obj["n"]) == lock:
                    if kind == "header":
                        if header is None:
                            header = obj
                            print(
                                f"Header: {header['name']!r}  "
                                f"{header['olen']:,} bytes  "
                                f"({header['zlen']:,} compressed, "
                                f"{header['algo']})"
                            )
                    else:
                        before = len(decoder.recovered)
                        decoder.add(obj["seed"], obj["droplet"])

                        if len(decoder.recovered) > before:
                            last_new = time.time()
                            print(
                                f"\rChunks {len(decoder.recovered)}"
                                f"/{decoder.n}  "
                                f"(droplets {decoder.droplets_used})",
                                end="",
                                flush=True,
                            )

                    if points is not None:
                        pts = points.astype(int).reshape(-1, 2)
                        cv2.polylines(frame, [pts], True, (0, 180, 0), 3)

                    if decoder.complete and header is not None:
                        print()
                        print("All chunks recovered.")
                        print("Verifying...")

                        out_path = finalize_received(
                            header, decoder, out_dir, overwrite,
                        )

                        print()
                        print("SHA-256 verified:")
                        print(header["sha"])
                        print()
                        if header["kind"] == KIND_BUNDLE:
                            print("Bundle extracted to:")
                        else:
                            print("Saved:")
                        print(out_path.resolve())
                        return

            draw_receiver_status(
                frame,
                len(decoder.recovered) if decoder else 0,
                decoder.n if decoder else None,
                decoder.droplets_used if decoder else 0,
                header is not None,
                lock[0] if lock else None,
            )

            if last_new and decoder and (time.time() - last_new > 5):
                cv2.putText(
                    frame,
                    "No new chunk for 5s: move closer / refocus / "
                    "lower sender FPS",
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
        reader.stop()
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

    ps = sub.add_parser(
        "send",
        help="Show one or more files as a looping QR stream "
             "(multiple files are sent as a tar bundle)",
    )
    ps.add_argument("files", type=Path, nargs="+")
    ps.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ps.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ps.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="Show the QR window on this monitor "
             "(1-based; see 'qrtx monitors')",
    )
    ps.add_argument(
        "--fullscreen",
        action="store_true",
        help="Fill the chosen monitor with the QR window",
    )

    pr = sub.add_parser("receive", help="Receive a QR stream from a camera")
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--out", type=Path, default=Path("received"))
    pr.add_argument("--width", type=int, default=1920)
    pr.add_argument("--height", type=int, default=1080)
    pr.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in --out instead of saving "
             "under a suffixed name",
    )
    pr.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="Show the preview window on this monitor "
             "(1-based; see 'qrtx monitors')",
    )

    sub.add_parser("monitors", help="List detected monitors and exit")

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "monitors":
        print_monitors()
        return

    require_runtime_deps()

    if args.cmd == "send":
        send_files(
            args.files, args.fps, args.chunk_size,
            args.monitor, args.fullscreen,
        )
    elif args.cmd == "receive":
        receive_file(
            args.camera, args.out, args.width, args.height, args.overwrite,
            args.monitor,
        )


if __name__ == "__main__":
    main()
