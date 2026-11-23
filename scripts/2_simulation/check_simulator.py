"""Pipeline 2 (simulation): Verify Push-T Simulation and Render Sample Rollout to MP4.

Verifies:
1. Pymunk physics engine dynamics and gym-pusht environment wrapper.
2. RGB observation extraction (96x96x3) and agent positioning.
3. Goal coverage tracking (intersection / goal area, as gym-pusht).
4. Video recorder with telemetry HUD overlay.
5. Exports a sample video to results/videos/sample_rollout.mp4 (config evaluation.video_dir).
"""

import os
import sys
# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import yaml
import numpy as np

from pusht_wm_dit.envs.pusht_sim import PushTSim, VideoRecorder, PushTExpertPolicy
from pusht_wm_dit.utils.metrics import compute_target_coverage_iou


def main():
    config_path = "configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_config = config["env"]
    eval_config = config["evaluation"]
    video_dir = eval_config["video_dir"]
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, "sample_rollout.mp4")

    print("=" * 70)
    print("STEP 2: TESTING PUSH-T SIMULATION & RENDERING SAMPLE ROLLOUT")
    print("=" * 70)

    sim = PushTSim(
        env_name=env_config["env_name"],
        render_size=tuple(env_config["render_size"]),
        max_steps=env_config["max_steps"],
    )
    recorder = VideoRecorder(output_path=video_path, fps=15, resolution=(384, 384))
    expert = PushTExpertPolicy()

    obs, info = sim.reset(seed=config["seed"])
    print(f"Environment reset successfully.")
    print(f"  Frame shape:       {obs['pixels'].shape} (dtype: {obs['pixels'].dtype})")
    print(f"  Agent position px: {obs['agent_pos_px']} | norm: {obs['agent_pos']}")
    print(f"  T-block pose:      {obs['block_pose']}")
    print(f"  Initial Coverage:  {obs['coverage'] * 100:.2f}%")

    # Record initial frame
    recorder.record_frame(
        frame=obs["pixels"],
        step=0,
        coverage=obs["coverage"],
        info_text="Initializing Simulation Rollout",
    )

    max_coverage = obs["coverage"]
    total_steps = 0

    print("\nRunning demonstration rollout with PushTExpertPolicy...")
    for step in range(1, 150):
        # Generate expert control action in workspace pixels [0, 512]
        act_cmd = expert.get_action(
            block_pose=obs["block_pose"],
            agent_pos=obs["agent_pos_px"],
        )

        # Step simulation (normalized_action=False because act_cmd is in pixel coordinates)
        obs, reward, terminated, truncated, info = sim.step(act_cmd, normalized_action=False)
        total_steps = step

        cov = obs["coverage"]
        if cov > max_coverage:
            max_coverage = cov

        recorder.record_frame(
            frame=obs["pixels"],
            step=step,
            coverage=cov,
            info_text="Expert Push Control",
        )

        if step % 25 == 0 or terminated or truncated:
            print(
                f"  Step {step:03d} | Coverage: {cov * 100:5.1f}% | "
                f"T-block: [{obs['block_pose'][0]:.1f}, {obs['block_pose'][1]:.1f}] | "
                f"Agent: [{obs['agent_pos_px'][0]:.1f}, {obs['agent_pos_px'][1]:.1f}]"
            )

        if terminated or truncated:
            break

    sim.close()
    recorder.save()

    print("\nSimulation Rollout Results:")
    print(f"  Total Steps:      {total_steps}")
    print(f"  Max Coverage:     {max_coverage * 100:.2f}%")
    print(f"  Final Coverage:   {obs['coverage'] * 100:.2f}%")
    print(f"  Video Output:     {video_path} (Size: {os.path.getsize(video_path):,} bytes)")
    print("=" * 70)
    print("STEP 2: SIMULATION & RENDERING VERIFICATION COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
