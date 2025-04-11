import numpy as np
import jax


def calc_return_to_go(rewards, terminals, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    A config dict for getting the default high/low rewrd values for each envs
    """
    if len(rewards) == 0:
        return np.array([])
    
    if is_sparse_reward:
        reward_neg = reward_neg * reward_scale + reward_bias
    else:
        assert not is_sparse_reward, "If you want to try on a sparse reward env, please add the reward_neg value in the ENV_CONFIG dict."

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg): 
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1-gamma))] * len(rewards)
    else:
        return_to_go = [0] * len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            return_to_go[-i-1] = rewards[-i-1] + reward_neg * prev_return * (1 - terminals[-i-1])
            prev_return = return_to_go[-i-1]

    return np.array(return_to_go, dtype=np.float32)


def add_mc_returns_to_trajectory(trajectory, gamma, reward_scale, reward_bias, reward_neg, is_sparse_reward):
    """
    undate every transition in the trajectory and add mc_returns
    return the updated trajectory
    """
    rewards = [t['rewards'] for t in trajectory]
    terminals = [t['dones'] for t in trajectory]

    mc_returns = calc_return_to_go(
        rewards=rewards,
        terminals=terminals,
        gamma=gamma,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        reward_neg=reward_neg, 
        is_sparse_reward=is_sparse_reward,
    )

    for i, transition in enumerate(trajectory):
        transition['mc_returns'] = mc_returns[i]

    return trajectory


def add_embeddings_to_trajectory(trajectory, model, tasks):
    """
    undate every transition in the trajectory and add embeddings
    return the updated trajectory
    """
    for i in range(len(trajectory)):
        observation = trajectory[i]['observations']
        
        image_primary = observation["side_policy_256"]
        image_wrist = observation["wrist_1"]
        # Add batch dimension
        image_primary = image_primary[np.newaxis, ...]
        image_wrist = image_wrist[np.newaxis, ...]
        timestep_pad_mask =  np.array([[True, True]])
        
        observation = {"image_primary": image_primary,
            "image_wrist": image_wrist, 
            "timestep_pad_mask": timestep_pad_mask,
        }
        
        action_embeddings = model.sample_transformer(observation, tasks,)
        # Now, action_embeddings is (batch_size, window_size, embedding_size)
        
        action_embeddings = action_embeddings[:, -1, :] # remove window_size dimension
        
        trajectory[i]['embeddings'] = action_embeddings

    return trajectory

def add_next_embeddings_to_trajectory(trajectory):
    """
    undate every transition in the trajectory and add next_embeddings
    return the updated trajectory
    """
    for i in range(len(trajectory)):
        if i == len(trajectory) - 1:
            trajectory[i]['next_embeddings'] = trajectory[i]['embeddings']
        else:
            trajectory[i]['next_embeddings'] = trajectory[i+1]['embeddings']

    return trajectory


