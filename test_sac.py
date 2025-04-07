import os

import imageio
import numpy as np
import torch
from PIL import Image

from homework3 import Hw3Env
from SAC import SACAgent
from state_processing import enhance_state


def process_frame(frame):
    """Convert frame to a format suitable for GIF saving"""
    if isinstance(frame, torch.Tensor):

        if len(frame.shape) == 3:
            frame = frame.permute(1, 2, 0).cpu().numpy()
        else:
            frame = frame.cpu().numpy()

    if len(frame.shape) < 2 or (len(frame.shape) == 3 and frame.shape[2] > 4):

        vis_size = 256
        img = np.ones((vis_size, vis_size, 3), dtype=np.uint8) * 240

        center = vis_size // 2
        cv_radius = 3
        img[
            center - cv_radius : center + cv_radius,
            center - cv_radius : center + cv_radius,
        ] = [0, 0, 255]

        return img

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    return frame


def test_agent(agent, env, num_episodes=10, save_gif=True):
    """Test the agent for a number of episodes with minimal logging."""
    all_rewards = []
    all_min_distances = []
    all_close_to_goal = []
    all_frames = []

    for episode in range(num_episodes):
        env.reset()

        observation = env.high_level_state()
        enhanced_obs = enhance_state(observation)

        episode_reward = 0
        done = False
        min_obj_goal_dist = float("inf")
        very_close_to_goal = False
        episode_frames = []

        print(f"Testing episode {episode+1}/{num_episodes}")

        while not done:

            if save_gif:
                raw_frame = env.state()
                processed_frame = process_frame(raw_frame)
                episode_frames.append(processed_frame)

            action = agent.act(enhanced_obs, deterministic=True)

            action_tensor = torch.FloatTensor(
                action.tolist() if isinstance(action, np.ndarray) else [action]
            )
            _, reward, done, truncated = env.step(action_tensor)
            done = done or truncated

            observation = env.high_level_state()
            enhanced_obs = enhance_state(observation)

            obj_pos = observation[2:4]
            goal_pos = observation[4:6]
            obj_goal_dist = np.linalg.norm(np.array(obj_pos) - np.array(goal_pos))
            min_obj_goal_dist = min(min_obj_goal_dist, obj_goal_dist)
            if obj_goal_dist < 0.01:
                very_close_to_goal = True

            episode_reward += reward

        all_rewards.append(episode_reward)
        all_min_distances.append(min_obj_goal_dist)
        all_close_to_goal.append(1 if very_close_to_goal else 0)

        if save_gif and episode_frames:
            all_frames.append(episode_frames)

        print(f"  Reward: {episode_reward:.2f}, Min Distance: {min_obj_goal_dist:.4f}")

    print("\nTest Results Summary:")
    print(f"  Average Reward: {np.mean(all_rewards):.4f}")
    print(f"  Average Min Distance to Goal: {np.mean(all_min_distances):.4f}")

    return {
        "rewards": all_rewards,
        "min_distances": all_min_distances,
        "frames": all_frames,
    }


def save_as_gif(frames, path, fps=30):
    """Save frames as GIF with proper error handling"""
    try:

        pil_frames = []
        for frame in frames:
            try:
                pil_img = Image.fromarray(frame)
                pil_frames.append(pil_img)
            except Exception as e:
                print(f"Error converting frame to PIL image: {e}")
                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
                continue

        if pil_frames:
            pil_frames[0].save(
                path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=True,
                duration=1000 // fps,
                loop=0,
            )
            return True
        return False
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False


if __name__ == "__main__":

    model_path = "saved_models_sac_2/sac_checkpoint_9000.pt"
    episodes = 1

    env = Hw3Env(render_mode="offscreen")
    raw_observation = env.high_level_state()
    enhanced_observation = enhance_state(raw_observation)
    observation_dim = enhanced_observation.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        observation_dim=observation_dim, action_dim=2, hidden_dim=256, device=device
    )

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
    else:
        print(f"Model file {model_path} not found!")
        exit(1)

    results = test_agent(agent, env, episodes)

    os.makedirs("gifs", exist_ok=True)
    for i, frames in enumerate(results["frames"]):
        if frames:
            gif_path = f"gifs/episode_{i+1}.gif"
            print(f"Saving GIF to {gif_path}")

            if save_as_gif(frames, gif_path):
                print(f"  Successfully saved GIF: {gif_path}")
            else:
                print(f"  Failed to save GIF: {gif_path}")
