#!/usr/bin/env python3
"""Test multi-view rendering: 16 fixed + 16 moving views for OvenFactory seed 0.

Distributes 32 views across 4 GPUs (8 per GPU).
Each GPU runs one Blender process that renders its assigned views sequentially.
"""
import os, subprocess, time, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
BASE_DIR = "/mnt/data/yurh/Infinite-Mobility"
NUM_GPUS = 4

# 16 fixed hemisphere views
HEMI_VIEWS = [f"hemi_{i:02d}" for i in range(16)]

# 8 orbit views (back-to-front) + 8 sweep views (front hemisphere)
ORBIT_VIEWS = [f"orbit_{i:02d}" for i in range(8)]
SWEEP_VIEWS = [f"sweep_{i:02d}" for i in range(8)]
MOVING_VIEWS = ORBIT_VIEWS + SWEEP_VIEWS


def render_batch(factory, seed, animode, static_views, moving_views, gpu_id, out_dir):
    """Run one Blender process to render a batch of views."""
    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", "render_articulation.py", "--",
        "--factory", factory, "--seed", str(seed), "--device", "0",
        "--output_dir", out_dir, "--resolution", "512", "--samples", "32",
        "--duration", "4.0", "--fps", "30", "--animode", str(animode),
        "--skip_bg",
    ]
    if static_views:
        cmd += ["--views"] + static_views
    else:
        # Must provide at least one view; use a dummy that won't match
        cmd += ["--views", "__none__"]
    if moving_views:
        cmd += ["--moving_views"] + moving_views

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env, capture_output=True, text=True, timeout=7200)
        dur = time.time() - t0
        # Count created mp4s
        n_mp4 = 0
        if os.path.isdir(out_dir):
            n_mp4 = sum(1 for f in os.listdir(out_dir) if f.endswith("_nobg.mp4"))
        ok = result.returncode == 0
        detail = ""
        if not ok:
            lines = result.stdout.strip().split("\n")[-5:]
            detail = "\n".join(lines)
        return gpu_id, ok, dur, n_mp4, len(static_views) + len(moving_views), detail
    except subprocess.TimeoutExpired:
        return gpu_id, False, time.time() - t0, 0, len(static_views) + len(moving_views), "TIMEOUT"


def main():
    factory = "OvenFactory"
    seed = 0
    animode = 0
    out_dir = os.path.join(BASE_DIR, "outputs", "animode_base_test", factory, str(seed))
    os.makedirs(out_dir, exist_ok=True)

    # Clean previous test outputs (hemi/orbit/sweep mp4s and frame dirs)
    for f in os.listdir(out_dir):
        fp = os.path.join(out_dir, f)
        if any(f.startswith(p) for p in ["hemi_", "orbit_", "sweep_"]):
            if os.path.isdir(fp):
                shutil.rmtree(fp)
            elif os.path.isfile(fp):
                os.remove(fp)

    # Distribute: 4 static + 4 moving per GPU
    jobs = []
    for gpu_id in range(NUM_GPUS):
        s_start = gpu_id * 4
        s_views = HEMI_VIEWS[s_start:s_start + 4]
        m_start = gpu_id * 4
        m_views = MOVING_VIEWS[m_start:m_start + 4]
        jobs.append((factory, seed, animode, s_views, m_views, gpu_id, out_dir))

    print(f"Rendering 32 views for {factory} seed {seed} animode {animode}")
    print(f"Output: {out_dir}\n")
    for _, _, _, sv, mv, gpu, _ in jobs:
        print(f"  GPU {gpu}: static={sv}, moving={mv}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as ex:
        futs = {}
        for job in jobs:
            fut = ex.submit(render_batch, *job)
            futs[fut] = job[5]  # gpu_id

        for fut in as_completed(futs):
            gpu_id, ok, dur, n_mp4, n_expected, detail = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] GPU {gpu_id}: {n_mp4}/{n_expected} videos, {dur:.0f}s")
            if detail:
                print(f"    {detail}")

    total = time.time() - t0
    # Count total mp4s
    all_mp4 = sorted(f for f in os.listdir(out_dir) if f.endswith("_nobg.mp4")
                     and any(f.startswith(p) for p in ["hemi_", "orbit_", "sweep_"]))
    print(f"\nDone! {len(all_mp4)}/32 videos in {total:.0f}s")
    print(f"Output: {out_dir}")
    for mp4 in all_mp4:
        print(f"  {mp4}")


if __name__ == "__main__":
    main()
