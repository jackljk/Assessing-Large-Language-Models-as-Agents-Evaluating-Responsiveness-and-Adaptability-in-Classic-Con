def text_to_action(text):
    """
        Given an output by the LLM in the form:
            Move Back revolute joint {value}, Back lower leg {value}, Front revolute joint {value}, Front lower leg {value}
        This function will return the corresponding action values for the environment 
    """
    action = [0, 0, 0, 0]
    split_text = text.split(", ")
    for i, action_value in enumerate(split_text):
        action[i] = float(action_value.split(" ")[-1])
    return action