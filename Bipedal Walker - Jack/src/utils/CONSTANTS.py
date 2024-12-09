ONE_ACTION = {
    "PRIMER": """
Imagine you are an expert at controlling a bipedal machine, and you have been given the task of controlling a bipedal machine in a reinforcement learning environment. The challenge is to walk a bipedal machine as far as possible without falling over. The machine has a hull, two legs, a back and front revolute joint, and lower leg. The machine has 10 lidar sensors that can detect the distance to objects in front of it. 

You will be given scenarios in the form of observations from the environment and the reward based on the last action and the Current Total Score. Your goal is to provide the next action in the following format:

'Move Back revolute joint `value`, Back lower leg 'value', Front revolute joint 'value', Front lower leg 'value''

The values must be in the range [-1, 1], and your goal is to maximize the reward and move the machine forward effectively. Only provide the action and no additional explanation.

Example
'Observation from last step: \nHull angle: -0.02\nAngular velocity: -0.03\nX velocity: -0.03\nY velocity: -0.01\nBack revolute joint angle: 0.48\nBack revolute joint speed: 1.00\nBack lower leg angle: 0.07\nBack lower leg speed: -1.00\nBack leg ground contact flag: 1.00\nFront revolute joint angle: 0.38\nFront revolute joint speed: 1.00\nFront lower leg angle: 0.08\nFront lower leg speed: -1.00\nFront leg ground contact flag: 1.00\nLidar 1 (0.00 rad): 0.45\nLidar 2 (0.15 rad): 0.45\nLidar 3 (0.30 rad): 0.47\nLidar 4 (0.45 rad): 0.50\nLidar 5 (0.60 rad): 0.54\nLidar 6 (0.75 rad): 0.61\nLidar 7 (0.90 rad): 0.72\nLidar 8 (1.05 rad): 0.90\nLidar 9 (1.20 rad): 1.00\nLidar 10 (1.35 rad): 1.00'
'The reward from the last step was: -0.25'
Score: -0.25

Action: Move Back revolute joint 0.5, Back lower leg -0.3, Front revolute joint 0.2, Front lower leg 0.7
""",
    "OBSERVATION_SKELETON": """
_
{obs_step}
{reward_step}
score: {score_step}
{additional_obs}

Action: 
""",
    "OBSERVATION_SKELETON_RESET": """
Now, given the following observation, rewards and score give me the next action for the following scenario:
_
{obs_step}
reward: None as it is the first step
score: 0
{additional_obs}

Action:
""",
}


FIVE_ACTIONS = {
    "PRIMER": """
Imagine you are an expert at controlling a bipedal machine, and you have been given the task of controlling a bipedal machine in a reinforcement learning environment. The challenge is to walk a bipedal machine as far as possible without falling over. The machine has a hull, two legs, a back and front revolute joint, and lower leg. The machine has 10 lidar sensors that can detect the distance to objects in front of it. 

You will be given scenarios in the form of observations from the environment and the reward based on the last action and the Current Total Score. Your goal is to provide the next action in the following format:

'Move Back revolute joint `value`, Back lower leg 'value', Front revolute joint 'value', Front lower leg 'value''

The values must be in the range [-1, 1], and your goal is to maximize the reward and move the machine forward effectively. Only provide the action and no additional explanation.

Example
'Observation from last step: \nHull angle: -0.02\nAngular velocity: -0.03\nX velocity: -0.03\nY velocity: -0.01\nBack revolute joint angle: 0.48\nBack revolute joint speed: 1.00\nBack lower leg angle: 0.07\nBack lower leg speed: -1.00\nBack leg ground contact flag: 1.00\nFront revolute joint angle: 0.38\nFront revolute joint speed: 1.00\nFront lower leg angle: 0.08\nFront lower leg speed: -1.00\nFront leg ground contact flag: 1.00\nLidar 1 (0.00 rad): 0.45\nLidar 2 (0.15 rad): 0.45\nLidar 3 (0.30 rad): 0.47\nLidar 4 (0.45 rad): 0.50\nLidar 5 (0.60 rad): 0.54\nLidar 6 (0.75 rad): 0.61\nLidar 7 (0.90 rad): 0.72\nLidar 8 (1.05 rad): 0.90\nLidar 9 (1.20 rad): 1.00\nLidar 10 (1.35 rad): 1.00'
rewards: [-0.25, 0.5, -0.75, 0.25, 0.1]
Score: -0.15

Actions
Move Back revolute joint 0.5, Back lower leg -0.3, Front revolute joint 0.2, Front lower leg 0.7
Move Back revolute joint 0.1, Back lower leg -0.9, Front revolute joint -0.2, Front lower leg 0.3
Move Back revolute joint -0.9, Back lower leg 0.8, Front revolute joint -0.1, Front lower leg -0.2
Move Back revolute joint 0.9, Back lower leg -0.1, Front revolute joint 0.6, Front lower leg 0.4
Move Back revolute joint -0.1, Back lower leg -0.9, Front revolute joint 0.1, Front lower leg 0.9
""",
    "OBSERVATION_SKELETON": """
_
{obs_step}
rewards: {reward_step}
score: {score_step}
{additional_obs}

Actions 
""",
    "OBSERVATION_SKELETON_RESET": """
Now, given the following observation, rewards and score give me the optimal best 5 action for the following scenario:
_
{obs_step}
reward: None as it is the first step
score: 0
{additional_obs}

Actions
""",
}


LLMS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}