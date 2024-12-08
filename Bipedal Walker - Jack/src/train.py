import gymnasium as gym
import time
from omegaconf import DictConfig, OmegaConf
import openai

from utils.action import text_to_action
from utils.observations import obs_to_text
from utils.rewards import reward_to_text, rewards_to_text
from utils.loggers import save_obs, store_to_csv
from llms.gpt import get_action
from experiments.enhanced import enhanced_observation
import os
import glob

import logging

# A logger for this file
log = logging.getLogger(__name__)


def train(
    env: gym.Env,
    client: openai.OpenAI,
    cfg: DictConfig,
):
    # Trainer Constants
    PRIMER_STARTER_LENGTH = (
        cfg.trainer.observation_text_length
    )  # Should be determined at runtime (TODO)
    OBS_TEXT_LENGTH = (
        cfg.trainer.primer_length
    )  # Should be determined at runtime (TODO)
    NUMBER_OF_OBS_MEMORY = cfg.trainer.observation_memory_size
    NO_ACTIONS = cfg.trainer.no_actions

    # Init variables
    steps = cfg.trainer.no_steps
    delay = cfg.trainer.delay

    # Get primers for experiment
    primer_dict = _determine_primers(NO_ACTIONS)

    # Get the skeleton for the observations
    primer = primer_dict["PRIMER"]
    obs_skeleton = primer_dict["OBSERVATION_SKELETON"]
    obs_skeleton_reset = primer_dict["OBSERVATION_SKELETON_RESET"]

    # Train variables to keep track of
    score, rewards, done, store = 0, [], False, {}
    action_text_store, action_store = [], []
    obs, _ = env.reset()

    # Variables to keep track of for the enhanced observations
    score_history, back_leg_movement_history, front_leg_movement_history = [], [], []

    # Make first primer
    obs_text = obs_to_text(obs)
    primer += obs_skeleton_reset.format(obs_step=obs_text, additional_obs="")

    for i in range(steps):
        log.info(f"Step: {i + 1}")
        # sleep to not exceed the rate limit
        time.sleep(delay)

        # Get action from gpt
        response = get_action(primer, client)
        action_texts = response.split("\n")
        for action_text in action_texts:
            # Get each of the actions given by the LLM
            action = text_to_action(action_text)
            log.info(f"Executing action: {action}")
            # perform action
            obs, reward, done, _, _ = env.step(action)

            # Update score
            score += reward

            # save the reward
            rewards.append(reward)
            # save the action
            action_text_store.append(action_text)
            action_store.append(action)

            # Add the action to the primer
            primer += action_text + "\n"

            if done:
                log.info(
                    "The bipedal machine has fallen over/took too long. The total score was: ",
                    score,
                )
                break

        # for enhance observations
        if cfg.experiment.type == "enhanced":
            enhanced_obs_text, back_leg_movement_history, front_leg_movement_history = (
                enhanced_observation(
                    obs,
                    cfg,
                    score_history,
                    back_leg_movement_history,
                    front_leg_movement_history,
                )
            )
            # Update the history for enhanced observations
            score_history.append(score)

        # Generate new primer
        obs_text = obs_to_text(obs)
        if NO_ACTIONS > 1:
            reward_text = rewards_to_text(rewards)  # For multiple actions
        else:
            reward_text = reward_to_text(reward)

        if cfg.experiment.type == "enhanced":
            obs_step = obs_skeleton.format(
                obs_step=obs_text,
                reward_step=reward_text,
                score_step=score,
                additional_obs=enhanced_obs_text,
            )
        else:
            obs_step = obs_skeleton.format(
                obs_step=obs_text,
                reward_step=reward_text,
                score_step=score,
                additional_obs="",
            )

        primer += obs_step  # actions are already added

        # Save the observations
        log.info("Saving observations")
        save_obs(
            obs,
            reward,
            score,
            i,
            action_text_store,
            action_store,
            store,
            obs_skeleton,
            enhanced_obs_text,
        )

        # reset rewards
        rewards = []
        action_text_store = []
        action_store = []

        # Check length of primer
        if len(primer) > PRIMER_STARTER_LENGTH + OBS_TEXT_LENGTH * NUMBER_OF_OBS_MEMORY:
            # split at "_" get the primer front and the continuation
            log.info("Primer too long, splitting")
            primer_splits = primer.split("_")
            primer = (
                primer_splits[0] + "_" + "".join(primer_splits[2:])
            )  # Remove the first observation and reward from memory

    # TODO: When steps are done or break out of episode, restart and start training with the next episode (May not be able to implement due to rate limit constraints so finetuning may not be possible with the given resources)

    log.info("Episode done, score: {score}")
    log.info("Storing observations")
    store_to_csv(store, cfg)


def _determine_primers(
    no_actions: int,
):
    log.info(f"Determining primers for {no_actions} actions")
    if no_actions == 1:
        from utils.CONSTANTS import ONE_ACTION

        return ONE_ACTION
    elif no_actions == 5:
        from utils.CONSTANTS import FIVE_ACTIONS

        return FIVE_ACTIONS
    else:
        raise ValueError("Number of actions not supported")
