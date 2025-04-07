import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter

from REINFORCE import Agent
from state_processing import calculate_reward

if __name__ == "__main__":
    env = gym.make("Pusher-v5", max_episode_steps=50)

    action_dim = env.action_space.shape[0]

    observation_dim = 9

    hidden_dim = 256
    gamma = 0.9
    lr = 3e-4
    std_factor = 0.9999
    min_std = 0.25

    N_EPISODE = 1000000
    TEST_FREQ = 10000
    test_episode = 0
    N_TEST_EPISODE = 10
    render_test = False

    agent = Agent(
        observation_dim,
        action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        lr=lr,
        entropy_coef=0.1,
    )
    writer = SummaryWriter()

    for episode in range(1, N_EPISODE + 1):
        done = False
        observation, _ = env.reset()
        ep_reward = 0
        fingertip_start_idx = 14
        object_start_idx = 17
        goal_start_idx = 20

        x_ee, y_ee, z_ee = observation[fingertip_start_idx : fingertip_start_idx + 3]
        x_obj, y_obj, z_obj = observation[object_start_idx : object_start_idx + 3]
        x_goal, y_goal, z_goal = observation[goal_start_idx : goal_start_idx + 3]

        while not done:

            action = agent.act(observation[14:], deterministic=False)
            observation, reward, done, truncated, _ = env.step(action=action)
            fingertip_start_idx = 14
            object_start_idx = 17
            goal_start_idx = 20

            next_x_ee, next_y_ee, z_ee = observation[
                fingertip_start_idx : fingertip_start_idx + 3
            ]
            next_x_obj, next_y_obj, z_obj = observation[
                object_start_idx : object_start_idx + 3
            ]
            next_x_goal, next_y_goal, z_goal = observation[
                goal_start_idx : goal_start_idx + 3
            ]
            done = done or truncated

            ep_reward += reward

            reward, _ = calculate_reward(
                current_ee_pos=[x_ee, y_ee],
                current_goal_pos=[x_goal, y_goal],
                current_obj_pos=[x_obj, y_obj],
                next_ee_pos=[next_x_ee, next_y_ee],
                next_goal_pos=[next_x_goal, next_y_goal],
                next_obj_pos=[next_x_obj, next_y_obj],
            )
            x_ee, y_ee, z_ee = observation[
                fingertip_start_idx : fingertip_start_idx + 3
            ]
            x_obj, y_obj, z_obj = observation[object_start_idx : object_start_idx + 3]
            x_goal, y_goal, z_goal = observation[goal_start_idx : goal_start_idx + 3]
            agent.collect(reward=reward, log_prob=None)

        print(f"Episode {episode} = Reward: {ep_reward}")
        fingertip_start_idx = 14
        object_start_idx = 17
        goal_start_idx = 20

        x_ee, y_ee, z_ee = observation[fingertip_start_idx : fingertip_start_idx + 3]
        x_obj, y_obj, z_obj = observation[object_start_idx : object_start_idx + 3]
        x_goal, y_goal, z_goal = observation[goal_start_idx : goal_start_idx + 3]

        ee_obj_dist = ((x_ee - x_obj) ** 2 + (y_ee - y_obj) ** 2) ** 0.5
        obj_goal_dist = ((x_obj - x_goal) ** 2 + (y_obj - y_goal) ** 2) ** 0.5

        object_at_goal = obj_goal_dist < 0.05

        print("Distance: ", obj_goal_dist)

        print("Goal reached!!") if object_at_goal else None

        writer.add_scalar("Train", ep_reward, episode)

        agent.learn()

        if episode % TEST_FREQ == 0:
            test_rewards = []
            for _ in range(N_TEST_EPISODE):
                done = False
                observation, _ = env.reset()
                test_episode += 1
                ep_reward = 0
                while not done:
                    action = agent.act(observation, deterministic=True)
                    observation, reward, done, truncated, _ = env.step(action)
                    done = done or truncated
                    ep_reward += reward

                    if render_test:
                        frame = env.render()
                        plt.clf()
                        plt.imshow(frame)
                        plt.axis("off")
                        plt.pause(0.2)
                    print("Test Episode: ", test_episode, "Took Action: ", action)
                print(f"Test Episode {test_episode} = Reward: {ep_reward}")
                test_rewards.append(ep_reward)

            avg_test_reward = sum(test_rewards) / len(test_rewards)
            writer.add_scalar("Test", avg_test_reward, episode)

            if render_test:
                plt.close()

    writer.close()
