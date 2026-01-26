#!/usr/bin/env python
"""
Multi-seed evaluation wrapper for π0 remote policy inference.

Runs multiple episodes with different seeds and collects statistics.

Usage:
  python scripts/pi0/run_pi0_remote_multi_seed.py \
    --env-id StackCube-v2 \
    --server ws://0.0.0.0:8000 \
    --num-seeds 10 \
    --start-seed 0 \
    --save-videos
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run π0 remote policy inference with multiple seeds"
    )
    
    # Multi-seed specific args
    p.add_argument("--num-seeds", type=int, default=10, help="Number of seeds to test")
    p.add_argument("--start-seed", type=int, default=0, help="Starting seed value")
    p.add_argument("--save-videos", action="store_true", help="Save videos for all runs")
    p.add_argument("--save-trajectories", action="store_true", help="Save trajectory.npz for all runs")
    
    # Common inference args
    p.add_argument("--server", type=str, required=True, help="e.g. ws://127.0.0.1:8000")
    p.add_argument("--env-id", type=str, default="StackCube-v2")
    p.add_argument("--obs-mode", type=str, default="rgb")
    p.add_argument("--control-mode", type=str, default="pd_ee_delta_pose")
    p.add_argument("--render-mode", type=str, default=None, help="e.g. 'human' for visualization")
    p.add_argument("--max-steps", type=int, default=200)
    
    # Observation keys
    p.add_argument("--image-key", type=str, default="sensor_data/base_camera/rgb")
    p.add_argument("--wrist-image-key", type=str, default="sensor_data/hand_camera/rgb")
    p.add_argument("--state-keys", type=str, nargs="+", default=None)
    p.add_argument("--prompt", type=str, default=None)
    
    # Output
    p.add_argument("--output-root", type=str, default="outputs/pi0_multi_seed")
    
    args = p.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{args.env_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare base command
    repo_root = Path(__file__).resolve().parents[2]
    run_script = repo_root / "scripts" / "pi0" / "run_pi0_remote.py"
    
    results = []
    success_count = 0
    
    print("=== Multi-Seed Evaluation ===")
    print(f"Environment: {args.env_id}")
    print(f"Server: {args.server}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.num_seeds - 1}")
    print(f"Output: {output_dir}")
    print()
    
    for i in range(args.num_seeds):
        seed = args.start_seed + i
        seed_output = output_dir / f"seed_{seed:03d}"
        
        print(f"[{i+1}/{args.num_seeds}] Running seed {seed}...", end=" ", flush=True)
        
        # Build command
        cmd = [
            sys.executable,
            str(run_script),
            "--server", args.server,
            "--env-id", args.env_id,
            "--obs-mode", args.obs_mode,
            "--control-mode", args.control_mode,
            "--seed", str(seed),
            "--max-steps", str(args.max_steps),
            "--image-key", args.image_key,
            "--wrist-image-key", args.wrist_image_key,
            "--output-root", str(seed_output.parent),
        ]
        
        if args.state_keys:
            cmd += ["--state-keys"] + args.state_keys
        
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        
        if args.render_mode:
            cmd += ["--render-mode", args.render_mode]
        
        if args.save_videos:
            cmd += ["--save-video"]
        
        if args.save_trajectories:
            cmd += ["--save-trajectory"]
        
        # Run
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )
            
            # Parse result
            success = False
            steps = 0
            for line in result.stdout.splitlines():
                if "done:" in line and "'success':" in line:
                    success = "True" in line
                if "'steps':" in line:
                    try:
                        steps = int(line.split("'steps':")[1].split(",")[0].strip())
                    except Exception:
                        pass
            
            results.append({
                "seed": seed,
                "success": success,
                "steps": steps,
            })
            
            if success:
                success_count += 1
                print(f"✓ Success ({steps} steps)")
            else:
                print(f"✗ Failed ({steps} steps)")
                
        except subprocess.TimeoutExpired:
            print("✗ Timeout")
            results.append({
                "seed": seed,
                "success": False,
                "steps": args.max_steps,
            })
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "seed": seed,
                "success": False,
                "steps": 0,
            })
    
    # Summary
    print()
    print("=== Summary ===")
    print(f"Total runs: {args.num_seeds}")
    print(f"Success: {success_count}")
    print(f"Failure: {args.num_seeds - success_count}")
    print(f"Success rate: {100 * success_count / args.num_seeds:.1f}%")
    
    if results:
        avg_steps = sum(r["steps"] for r in results) / len(results)
        print(f"Average steps: {avg_steps:.1f}")
    
    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write(f"Environment: {args.env_id}\n")
        f.write(f"Server: {args.server}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\nResults:\n")
        for r in results:
            f.write(f"  Seed {r['seed']}: {'✓' if r['success'] else '✗'} ({r['steps']} steps)\n")
        f.write("\nSummary:\n")
        f.write(f"  Success rate: {100 * success_count / args.num_seeds:.1f}%\n")
        f.write(f"  Average steps: {avg_steps:.1f}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
