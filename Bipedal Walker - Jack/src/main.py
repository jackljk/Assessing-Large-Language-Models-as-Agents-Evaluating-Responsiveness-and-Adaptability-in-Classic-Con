import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import openai
from train import train
import time

import logging

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="defaults")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Init environment
    env = gym.make(cfg.env.environment, render_mode=cfg.env.render_mode)
    obs, _ = env.reset()
    log.info(f"Environment initialized")

    # Start time counter
    start_time = time.time()

    # Init client
    client = openai.OpenAI()
    log.info(f"LLM Client initialized")

    log.info(
        f"Starting training for {cfg.trainer.no_steps} steps and {cfg.trainer.no_episodes}. "
    )
    log.info(f"Delay between steps: {cfg.trainer.delay} seconds")
    log.info(f"Experiment type: {cfg.experiment.type}")
    # Train
    train(env, client, cfg)

    # End time counter
    end_time = time.time()
    elapsed_time = end_time - start_time
    log.info(f"Training took {elapsed_time:.2f} seconds")
    log.info(f"Training completed")


if __name__ == "__main__":
    main()
