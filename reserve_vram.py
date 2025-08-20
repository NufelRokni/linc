#!/usr/bin/env python3
"""
reserve_vram.py

Pre-allocate (reserve) most free VRAM on all CUDA-visible GPUs to deter other
processes from grabbing memory right before we launch the main job.

Environment variables:
- RESERVE_MARGIN_MB: Per-GPU free memory to leave unallocated (default: 200 MB)
- RESERVE_HOLD_SEC: How long to hold the reservation before exiting (default: 2 s)

Notes:
- This runs as a separate process. Memory is released when this script exits.
- Use a small hold time to reduce the race window before the main job starts
  and allocates its own memory.
- If you make the hold time long, it can starve your own job of memory since it
  runs in a separate process. Use with care.
"""
import os
import sys
import time

try:
    import torch
except Exception as e:
    print(f"[reserve_vram] PyTorch not available: {e}", file=sys.stderr)
    sys.exit(0)

if not torch.cuda.is_available():
    print("[reserve_vram] CUDA not available; nothing to reserve.")
    sys.exit(0)

margin_mb = int(os.environ.get("RESERVE_MARGIN_MB", "200"))
hold_sec = float(os.environ.get("RESERVE_HOLD_SEC", "2"))

handles = []
ngpu = torch.cuda.device_count()

for i in range(ngpu):
    dev = torch.device(f"cuda:{i}")
    try:
        torch.cuda.set_device(dev)
        free, total = torch.cuda.mem_get_info()
        reserve_bytes = int(free) - margin_mb * 1024 * 1024
        if reserve_bytes <= 0:
            print(f"[reserve_vram] {dev}: not enough free memory to reserve (free={free/1024**2:.1f} MB). Skipping.")
            continue
        n_float32 = reserve_bytes // 4
        t = torch.empty(n_float32, dtype=torch.float32, device=dev)
        handles.append(t)
        print(f"[reserve_vram] {dev}: reserved ~{reserve_bytes/1024**2:.1f} MB (leave {margin_mb} MB).")
    except RuntimeError as e:
        print(f"[reserve_vram] {dev}: failed to reserve memory: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[reserve_vram] {dev}: unexpected error: {e}", file=sys.stderr)

if handles:
    try:
        print(f"[reserve_vram] Holding reservation for {hold_sec} seconds…")
        time.sleep(hold_sec)
    except KeyboardInterrupt:
        pass
else:
    print("[reserve_vram] No reservations made.")

# Release by exiting
print("[reserve_vram] Releasing reservations and exiting.")
