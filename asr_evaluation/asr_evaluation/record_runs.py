"""Record N normal driving runs (Phase 1): repeatedly launches the headless
fast-physics recording launch, one rosbag per run.

    ros2 run asr_evaluation record_runs -- --n 10 --duration 90 --out data/raw
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import time


# world -> (fast world file, spawn x, spawn y); spawn poses follow the
# turtlebot3_gazebo launch defaults for each world
WORLDS = {
    "tb3_world": ("tb3_world_fast.world", -2.0, -0.5),
    "house": ("house_fast.world", -2.0, -0.5),
    "stage4": ("stage4_fast.world", 0.0, 0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--start", type=int, default=0, help="first run index/seed")
    ap.add_argument("--world", default="tb3_world", choices=list(WORLDS),
                    help="world name; bag dirs are named <world>_<i>")
    args = ap.parse_args()
    world_file, x, y = WORLDS[args.world]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.start, args.start + args.n):
        bag = out / f"{args.world}_{i:02d}"
        if bag.exists():
            print(f"skip existing {bag}")
            continue
        tmp = bag.with_suffix(".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        print(f"=== recording run {i} -> {bag}")
        cmd = ["ros2", "launch", "asr_bringup", "record.launch.py",
               f"world:={world_file}", f"x_pose:={x}", f"y_pose:={y}",
               f"drive_seed:={1000 * list(WORLDS).index(args.world) + i}",
               f"drive_duration:={args.duration}", f"out:={tmp}"]
        for attempt in range(2):  # gzserver occasionally dies on startup
            subprocess.run(["pkill", "-x", "gzserver"], check=False)
            time.sleep(2.0)
            if tmp.exists():
                shutil.rmtree(tmp)
            try:  # wall-time guard: fast physics usually finishes earlier
                subprocess.run(cmd, timeout=args.duration * 3 + 120, check=False)
            except subprocess.TimeoutExpired:
                print(f"run {i}: launch timed out, killing", file=sys.stderr)
            subprocess.run(["pkill", "-x", "gzserver"], check=False)
            if tmp.exists() and sum(f.stat().st_size for f in tmp.iterdir()) > 1_000_000:
                break
            print(f"run {i}: attempt {attempt} produced no usable bag", file=sys.stderr)
        if tmp.exists():
            tmp.rename(bag)


if __name__ == "__main__":
    main()
