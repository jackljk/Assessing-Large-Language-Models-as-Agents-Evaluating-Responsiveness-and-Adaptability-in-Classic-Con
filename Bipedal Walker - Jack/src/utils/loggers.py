from utils.action import text_to_action
from utils.observations import obs_to_text
from utils.rewards import reward_to_text

import os
import glob
import pandas as pd
import logging

# A logger for this file
log = logging.getLogger(__name__)


def save_obs(
    obs,
    reward,
    score,
    step,
    actions_text,
    actions,
    store={},
    obs_skeleton=None,
    additional_obs="",
):
    obs_text = obs_to_text(obs)
    reward_text = reward_to_text(reward)
    score_text = f"Score: {score}"
    obs_skeleton_filled = obs_skeleton.format(
        obs_step=obs_text,
        reward_step=reward_text,
        score_step=score_text,
        additional_obs=additional_obs,
    )
    inner = {
        "obs": obs,
        "obs_text": obs_text,
        "reward": reward,
        "reward_text": reward_text,
        "score": score,
        "skeleton": obs_skeleton_filled,
        "actions_text": actions_text,
        "actions": actions,
    }
    store[step] = inner


def store_to_csv(store, cfg):
    """
    Store the observations in a csv file
    """
    experiment_type = cfg.experiment.type
    save_dir = cfg.logger.save_dir + "/" + experiment_type + "/" + cfg.experiment.llm + "/"

    # make a directory by the name of the experiment
    os.makedirs(save_dir, exist_ok=True)

    # get all the files in the directory
    files = glob.glob(save_dir + "*")
    file_name = f"{save_dir}observations_{len(files)}.csv"

    log.info(f"Storing observations in {file_name}")

    # create a dataframe
    df = pd.DataFrame(store).T
    df.to_csv(file_name)
