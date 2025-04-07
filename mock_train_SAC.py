import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from SAC import SACAgent

if __name__ == "__main__":
    env = gym.make(
        "InvertedDoublePendulum-v5", healthy_reward=10, render_mode="rgb_array"
    )

    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]

    seed = 42
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden_dim = 256
    gamma = 0.99
    lr = 3e-4
    tau = 0.005
    alpha = 0.2
    auto_entropy_tuning = True
    batch_size = 64
    replay_buffer_size = 1000

    N_EPISODE = 1000000
    TEST_FREQ = 10000
    LEARN_AFTER = 1000
    LEARN_FREQ = 1
    test_episode = 0
    N_TEST_EPISODE = 10
    render_test = False
    total_steps = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        auto_entropy_tuning=auto_entropy_tuning,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        device=device,
    )

    writer = SummaryWriter()

    for episode in range(1, N_EPISODE + 1):
        done = False
        observation, _ = env.reset()
        ep_reward = 0
        steps = 0

        while not done:

            action = agent.act(observation, deterministic=False)

            next_observation, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            agent.collect(observation, action, reward, next_observation, done)

            ep_reward += reward
            total_steps += 1
            steps += 1

            if total_steps > LEARN_AFTER and total_steps % LEARN_FREQ == 0:
                agent.learn()

            observation = next_observation

        print(
            f"Episode {episode} | Steps: {steps} | Reward: {ep_reward:.2f} | Total Steps: {total_steps}"
        )
        writer.add_scalar("Train_Mock/Reward", ep_reward, episode)
        writer.add_scalar("Train_Mock/Steps", steps, episode)

        if episode % TEST_FREQ == 0:
            test_rewards = []
            test_steps = []

            print("\n--- Testing ---")
            for i in range(N_TEST_EPISODE):
                done = False
                observation, _ = env.reset()
                test_episode += 1
                ep_reward = 0
                steps = 0

                while not done:

                    action = agent.act(observation, deterministic=True)

                    observation, reward, done, truncated, _ = env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    steps += 1

                    if render_test:
                        frame = env.render()
                        plt.clf()
                        plt.imshow(frame)
                        plt.axis("off")
                        plt.pause(0.01)

                print(
                    f"Test Episode {i+1}/{N_TEST_EPISODE} | Steps: {steps} | Reward: {ep_reward:.2f}"
                )
                test_rewards.append(ep_reward)
                test_steps.append(steps)

            avg_test_reward = sum(test_rewards) / len(test_rewards)
            avg_test_steps = sum(test_steps) / len(test_steps)
            writer.add_scalar("Test_Mock/AvgReward", avg_test_reward, episode)
            writer.add_scalar("Test_Mock/AvgSteps", avg_test_steps, episode)
            print(
                f"Test Summary | Avg Reward: {avg_test_reward:.2f} | Avg Steps: {avg_test_steps:.2f}"
            )
            print("--- Testing Complete ---\n")

            if render_test:
                plt.close()

    writer.close()
    env.close()
