#!/usr/bin/env python3
"""Inspect an H2 SONIC checkpoint so we can build the policy runner around it.

Downloads (or reads) the checkpoint files and prints, for every ONNX model,
the exact input/output tensor names, shapes, and dtypes, plus the contents of
any yaml/json config files. Paste the full output back to Claude -- the
observation layout it reveals dictates how the policy runner must be written.

Usage (either):
    python tools/inspect_h2_checkpoint.py --hf junsooki/h2_checkpoints
    python tools/inspect_h2_checkpoint.py --dir /path/to/downloaded/checkpoint

Requires: onnxruntime (pip install onnxruntime); for --hf also huggingface_hub.
"""

import argparse
import os


def inspect_onnx(path):
    import onnxruntime as ort

    print(f"\n=== ONNX: {path} ({os.path.getsize(path)/1e6:.1f} MB) ===")
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return
    print("  inputs:")
    for i in sess.get_inputs():
        print(f"    {i.name}: shape={i.shape} dtype={i.type}")
    print("  outputs:")
    for o in sess.get_outputs():
        print(f"    {o.name}: shape={o.shape} dtype={o.type}")
    # metadata sometimes carries the export config
    meta = sess.get_modelmeta()
    if meta.custom_metadata_map:
        print("  metadata:")
        for k, v in meta.custom_metadata_map.items():
            print(f"    {k}: {str(v)[:400]}")


def _summarize(obj, prefix="", depth=0, lines=None):
    import torch

    if lines is None:
        lines = []
    if len(lines) > 400 or depth > 4:
        return lines
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                lines.append(f"{prefix}{k}: Tensor{tuple(v.shape)} {v.dtype}")
            elif isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}: {type(v).__name__} ({len(v)} items)")
                _summarize(v, prefix + "  ", depth + 1, lines)
            else:
                lines.append(f"{prefix}{k}: {type(v).__name__} = {str(v)[:200]}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            lines.append(f"{prefix}[{i}]: {type(v).__name__} {str(v)[:120]}")
    return lines


def inspect_pt(path):
    print(f"\n=== PT CHECKPOINT: {path} ({os.path.getsize(path)/1e6:.1f} MB) ===")
    try:
        import torch
    except ImportError:
        print("  torch not installed -- pip install --user torch --index-url https://download.pytorch.org/whl/cpu")
        return
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  torch.load failed: {e}")
        return
    print(f"  top-level type: {type(ckpt).__name__}")
    for line in _summarize(ckpt):
        print("  " + line)


def dump_text(path, limit=8000):
    print(f"\n=== {path} ===")
    try:
        text = open(path, errors="replace").read()
        print(text[:limit])
        if len(text) > limit:
            print(f"... [{len(text)-limit} more chars truncated]")
    except Exception as e:
        print(f"  could not read: {e}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf", help="HuggingFace repo id, e.g. junsooki/h2_checkpoints")
    g.add_argument("--dir", help="local directory containing the checkpoint files")
    args = ap.parse_args()

    if args.hf:
        from huggingface_hub import snapshot_download

        root = snapshot_download(args.hf)
        print(f"downloaded to {root}")
    else:
        root = args.dir

    all_files = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            all_files.append(os.path.join(dirpath, f))

    print("\n=== FILE LISTING ===")
    for f in all_files:
        print(f"  {os.path.relpath(f, root)}  ({os.path.getsize(f)/1e6:.2f} MB)")

    for f in all_files:
        low = f.lower()
        if low.endswith(".onnx"):
            inspect_onnx(f)
        elif low.endswith((".yaml", ".yml", ".json", ".txt", ".md")):
            dump_text(f)
        elif low.endswith((".pt", ".pth", ".ckpt")):
            inspect_pt(f)


if __name__ == "__main__":
    main()
