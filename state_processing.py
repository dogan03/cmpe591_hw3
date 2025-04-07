import math

import numpy as np
import torch


def enhance_state(high_level_state, mock=False):
    """
    Optimized state representation for your 2D action space
    """
    if isinstance(high_level_state, torch.Tensor):
        high_level_state = high_level_state.cpu().numpy()

    if not mock:
        ee_pos = high_level_state[:2]
        obj_pos = high_level_state[2:4]
        goal_pos = high_level_state[4:6]
    else:
        ee_pos = high_level_state[:2]
        obj_pos = high_level_state[3:5]
        goal_pos = high_level_state[6:8]

    ee_to_obj = obj_pos - ee_pos
    obj_to_goal = goal_pos - obj_pos

    ee_obj_dist = np.linalg.norm(ee_to_obj) + 1e-6
    obj_goal_dist = np.linalg.norm(obj_to_goal) + 1e-6

    ee_to_obj_norm = ee_to_obj / ee_obj_dist
    obj_to_goal_norm = obj_to_goal / obj_goal_dist

    enhanced_state = np.concatenate(
        [
            ee_pos,
            obj_pos,
            goal_pos,
            ee_to_obj_norm,
            obj_to_goal_norm,
            [ee_obj_dist],
            [obj_goal_dist],
        ]
    )

    return enhanced_state


def calculate_reward(
    current_ee_pos,
    current_obj_pos,
    current_goal_pos,
    next_ee_pos,
    next_obj_pos,
    next_goal_pos,
):
    """
    Reward function based on progress toward goals between current and next states
    """

    current_dist_ee_obj = np.linalg.norm(
        np.array(current_ee_pos) - np.array(current_obj_pos)
    )
    current_dist_obj_goal = np.linalg.norm(
        np.array(current_obj_pos) - np.array(current_goal_pos)
    )

    next_dist_ee_obj = np.linalg.norm(np.array(next_ee_pos) - np.array(next_obj_pos))
    next_dist_obj_goal = np.linalg.norm(
        np.array(next_obj_pos) - np.array(next_goal_pos)
    )

    ee_obj_progress = current_dist_ee_obj - next_dist_ee_obj
    obj_goal_progress = current_dist_obj_goal - next_dist_obj_goal

    obj_movement = np.linalg.norm(np.array(next_obj_pos) - np.array(current_obj_pos))

    ee_obj_reward = ee_obj_progress * 5.0

    obj_goal_reward = obj_goal_progress * 10.0

    movement_reward = 0.5 if obj_movement > 0.001 else 0.0

    goal_reward = 0.5 if next_dist_obj_goal < 0.01 else 0.0

    reward = (ee_obj_reward + obj_goal_reward + goal_reward) * 100

    if next_dist_obj_goal < 0.01:
        print("SUCCESS! Object reached goal")

    if next_dist_ee_obj < 0.02 and current_dist_ee_obj >= 0.02:
        print("CONTACT! End effector touching object")

    return reward, next_dist_obj_goal < 0.01
