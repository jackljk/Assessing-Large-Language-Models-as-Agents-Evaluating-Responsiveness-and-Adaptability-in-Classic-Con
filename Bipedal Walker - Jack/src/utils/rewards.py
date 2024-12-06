
def reward_to_text(reward):
    """
    Convert the reward to a human-readable string. For LLM experiments
    """
    return f"The reward from the last step was: {reward:.2f}"


def rewards_to_text(rewards):
    # Gets a list of rewards and returns a string with the rewards in the form "[reward1, reward2, reward3]"
    return "[" + ", ".join([str(reward) for reward in rewards]) + "]"