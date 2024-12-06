from utils.action import text_to_action
from utils.observations import obs_to_text
from utils.rewards import reward_to_text


def save_obs(obs, reward, score, step, actions_text, actions, store={}, obs_skeleton=None):
    obs_text = obs_to_text(obs)
    reward_text = reward_to_text(reward)
    score_text = f"Score: {score}"
    obs_skeleton_filled = obs_skeleton.format(obs_step=obs_text, reward_step=reward_text, score_step=score_text)
    inner = {
        "obs": obs,
        "obs_text": obs_text,
        "reward": reward,
        "reward_text": reward_text,
        "score": score,
        "skeleton": obs_skeleton_filled,
        "actions_text": actions_text,
        "actions": actions
    }
    store[step] = inner