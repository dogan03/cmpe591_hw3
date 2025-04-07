import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from homework3 import Hw3Env
from REINFORCE import Agent
from state_processing import calculate_reward, enhance_state

if __name__ == "__main__":
    env = Hw3Env(render_mode="offscreen")

    save_dir = "saved_models_reinforce_2"
    os.makedirs(save_dir, exist_ok=True)

    raw_observation = env.high_level_state()
    enhanced_observation = enhance_state(raw_observation)
    observation_dim = enhanced_observation.shape[0]

    print(f"Raw state dimension: {len(raw_observation)}")
    print(f"Enhanced state dimension: {observation_dim}")

    action_dim = 2

    hidden_dim = 128
    gamma = 0.99
    lr = 3e-4
    entropy_coef = 0.1

    N_EPISODE = 1000000
    TEST_FREQ = 100
    test_episode = 0
    N_TEST_EPISODE = 5
    render_test = False

    agent = Agent(
        observation_dim,
        action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        lr=lr,
        entropy_coef=entropy_coef,
    )
    writer = SummaryWriter()

    best_test_reward = float("-inf")
    best_test_close_rate = 0.0

    for episode in range(1, N_EPISODE + 1):
        env.n_splits = 30
        env.reset()
        done = False
        observation = env.high_level_state()
        enhanced_obs = enhance_state(observation)
        ep_reward = 0
        shaped_ep_reward = 0

        very_close_to_goal_train = False
        min_obj_goal_dist_train = float("inf")

        ee_pos = observation[:2]
        obj_pos = observation[2:4]
        goal_pos = observation[4:6]

        initial_ee_pos = ee_pos
        initial_obj_pos = obj_pos
        goal_pos = goal_pos

        while not done:

            action = agent.act(enhanced_obs, deterministic=False)

            action_tensor = torch.FloatTensor(
                action.tolist() if isinstance(action, np.ndarray) else [action]
            )

            _, env_reward, done, truncated = env.step(action_tensor)
            done = done or truncated

            next_observation = env.high_level_state()
            next_enhanced_obs = enhance_state(next_observation)

            next_ee_pos = next_observation[:2]
            next_obj_pos = next_observation[2:4]
            next_goal_pos = next_observation[4:6]

            obj_goal_dist = np.linalg.norm(
                np.array(next_obj_pos) - np.array(next_goal_pos)
            )
            min_obj_goal_dist_train = min(min_obj_goal_dist_train, obj_goal_dist)

            if obj_goal_dist < 0.01:
                very_close_to_goal_train = True

            shaped_reward, goal_reached = calculate_reward(
                current_ee_pos=ee_pos,
                current_goal_pos=goal_pos,
                current_obj_pos=obj_pos,
                next_ee_pos=next_ee_pos,
                next_goal_pos=next_goal_pos,
                next_obj_pos=next_obj_pos,
            )

            agent.collect(reward=shaped_reward, log_prob=None)

            ee_pos = next_ee_pos
            obj_pos = next_obj_pos
            goal_pos = next_goal_pos
            enhanced_obs = next_enhanced_obs

            ep_reward += env_reward
            shaped_ep_reward += shaped_reward

        writer.add_scalar("Train_Reinforce/Episode_Reward", ep_reward, episode)
        writer.add_scalar("Train_Reinforce/Shaped_Reward", shaped_ep_reward, episode)
        writer.add_scalar(
            "Train_Reinforce/Min_Obj_Goal_Distance", min_obj_goal_dist_train, episode
        )
        writer.add_scalar(
            "Train_Reinforce/Close_To_Goal",
            1.0 if very_close_to_goal_train else 0.0,
            episode,
        )

        agent.learn()

        if episode % 1000 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{episode}.pt")
            torch.save(
                {
                    "episode": episode,
                    "actor_state_dict": agent.actor.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint at episode {episode} to {checkpoint_path}")

        if episode % TEST_FREQ == 0:
            test_rewards = []
            close_to_goal_count = 0
            test_min_distances = []

            for test_ep in range(N_TEST_EPISODE):
                env.reset()
                env.n_splits = 30
                done = False
                observation = env.high_level_state()
                enhanced_obs = enhance_state(observation)
                test_episode += 1
                ep_reward = 0

                very_close_to_goal = False
                min_obj_goal_dist_test = float("inf")

                while not done:

                    action = agent.act(enhanced_obs, deterministic=True)

                    action_tensor = torch.FloatTensor(action)
                    _, reward, done, truncated = env.step(action_tensor)
                    done = done or truncated

                    observation = env.high_level_state()
                    obj_pos = observation[2:4]
                    goal_pos = observation[4:6]

                    obj_goal_dist = np.linalg.norm(
                        np.array(obj_pos) - np.array(goal_pos)
                    )
                    min_obj_goal_dist_test = min(min_obj_goal_dist_test, obj_goal_dist)

                    if obj_goal_dist < 0.01:
                        very_close_to_goal = True

                    enhanced_obs = enhance_state(observation)

                    ep_reward += reward

                    if render_test:
                        frame = env.render()
                        plt.clf()
                        plt.imshow(frame)
                        plt.axis("off")
                        plt.pause(0.2)

                    if test_ep == 0:
                        print(
                            f"Test Episode: {test_episode}, Action: [{action[0]:.4f}, {action[1]:.4f}], Obj-Goal Dist: {obj_goal_dist:.4f}"
                        )

                if very_close_to_goal:
                    close_to_goal_count += 1

                test_min_distances.append(min_obj_goal_dist_test)
                print(
                    f"Test Episode {test_episode} = Reward: {ep_reward}, Min Dist: {min_obj_goal_dist_test:.4f}, Close to goal: {very_close_to_goal}"
                )
                test_rewards.append(ep_reward)

            avg_test_reward = sum(test_rewards) / len(test_rewards)
            close_rate = close_to_goal_count / N_TEST_EPISODE
            avg_min_distance = sum(test_min_distances) / len(test_min_distances)

            writer.add_scalar("Test_Reinforce/Reward", avg_test_reward, episode)
            writer.add_scalar("Test_Reinforce/Close_To_Goal_Rate", close_rate, episode)
            writer.add_scalar(
                "Test_Reinforce/Avg_Min_Distance", avg_min_distance, episode
            )

            print(
                f"Test Summary: Avg Reward = {avg_test_reward:.4f}, Avg Min Dist = {avg_min_distance:.4f}, Close to Goal Rate = {close_rate:.2%}"
            )

            if avg_test_reward > best_test_reward:
                best_test_reward = avg_test_reward
                best_path = os.path.join(save_dir, "best_reward_model.pt")
                torch.save(
                    {
                        "episode": episode,
                        "actor_state_dict": agent.actor.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),
                        "test_reward": best_test_reward,
                    },
                    best_path,
                )
                print(
                    f"New best model saved with avg test reward: {best_test_reward:.4f}"
                )

            if close_rate > best_test_close_rate:
                best_test_close_rate = close_rate
                best_close_path = os.path.join(save_dir, "best_close_rate_model.pt")
                torch.save(
                    {
                        "episode": episode,
                        "actor_state_dict": agent.actor.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),
                        "close_rate": best_test_close_rate,
                    },
                    best_close_path,
                )
                print(
                    f"New best model saved with close to goal rate: {best_test_close_rate:.2%}"
                )

            if render_test:
                plt.close()

    writer.close()
