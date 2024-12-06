import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import openai
from train import train

STEPS = 500
OBS_TEXT_LENGTH = 800 # Approx
PRIMER_STARTER_LENGTH = 2000 # Approx
NUMBER_OF_OBS_MEMORY = 3 
save_dir = "runs/bipedal_walker/"
NUMBER_ACTIONS = 5


@hydra.main(version_base=None, config_path="configs/", config_name="defaults")
def main(
    cfg: DictConfig
):  
    print(OmegaConf.to_yaml(cfg))

    # Init environment
    env = gym.make(
        cfg.env.environment,
        render_mode=cfg.env.render_mode
    )
    obs, _ = env.reset()
    print("--Environment initialized--")
    
    # Init client
    client = openai.OpenAI()
    print("--Client initialized--")
    
    print("--Training started--")
    # Train
    train(
        env, 
        client,
        cfg.trainer.no_steps,
        cfg.trainer.delay,
        cfg.experiment.type,
        cfg
    )
    
    
if __name__ == "__main__":
    main()