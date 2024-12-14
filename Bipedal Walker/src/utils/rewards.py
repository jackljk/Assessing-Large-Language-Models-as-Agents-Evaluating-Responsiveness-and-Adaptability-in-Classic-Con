
def reward_to_text(reward):
    """
    Convert the reward to a human-readable string. For LLM experiments
    """
    return f"The reward from the last step was: {reward:.2f}"


def rewards_to_text(rewards):
    # Gets a list of rewards and returns a string with the rewards in the form "[reward1, reward2, reward3]"
    return "[" + ", ".join([str(reward) for reward in rewards]) + "]"


####################################################################################################
# The following functions are for enhancing the rewards
def reward_changes(scores, threshold, no_check=10):
    """
    Get the last x scores and check if there are any changes in the reward

    Args:
        scores: list of scores
    returns:
        string: message to the agent (False if there are no changes)
    """
    NO_CHANGE_IN_REWARD = """
    There has been no change in the reward for the last {no_check} steps. Please try a different strategy to maximize the reward and progress the Walker forward
    """ 
    
    GOOD_CHANGE_IN_REWARD = """
    There has been a positive change in the reward for the last {no_check} steps. Keep up the good work!
    """

    if len(scores) < no_check:
        return None

    last_x = scores[-no_check:]
    # find the absolute difference between the last x scores
    diff = [abs(last_x[i] - last_x[i + 1]) for i in range(len(last_x) - 1)]

    if all([d < threshold for d in diff]):
        return NO_CHANGE_IN_REWARD.format(no_check=no_check)
    else:
        return GOOD_CHANGE_IN_REWARD.format(no_check=no_check)