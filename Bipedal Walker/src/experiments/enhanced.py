from utils.CONSTANTS import *
from utils.observations import *
from utils.rewards import *
from utils.action import *


def enhanced_observation(
    obs,
    cfg,
    score_history=[],
    back_leg_movement_history=[],
    front_leg_movement_history=[],
):
    # Walker moving direction
    walker_moving_direction_text = walker_moving_direction(obs)

    # leg movement
    back_res, front_res, back_leg_movement_history, front_leg_movement_history = (
        leg_movement(
            obs,
            cfg.experiment.no_check,
            cfg.experiment.threshold_leg_movement,
            back_leg_movement_history,
            front_leg_movement_history,
        )
    )

    # Walker tilt
    walker_tilt_text = walker_tilting_forward(obs)

    # Reward changes
    reward_change_text = reward_changes(
        score_history,
        cfg.experiment.threshold_rewards_no_change,
        cfg.experiment.no_check,
    )

    # Handle if some of the enhanced observations are None (i.e not enough steps to bother checking)
    if back_res is None:
        # If back_res is None, then front_res is also None
        back_res = ""
        front_res = ""

    if reward_change_text is None:
        reward_change_text = ""

    output_text = f"""
    Additional Observations:
    {walker_moving_direction_text}
    {back_res}
    {front_res}
    {walker_tilt_text}
    {reward_change_text}
    """

    return output_text, back_leg_movement_history, front_leg_movement_history
