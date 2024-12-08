import gymnasium as gym

def obs_to_text(obs):
    """
    Convert the observation to a human-readable string. For LLM experiments.
    """
    starter = "Observation from last step: "
    hull_angle_speed = f"Hull angle: {obs[0]:.2f}"
    angular_velocity = f"Angular velocity: {obs[1]:.2f}"
    x_velocity = f"X velocity: {obs[2]:.2f}"
    y_velocity = f"Y velocity: {obs[3]:.2f}"
    back_revolute_joint_angle = f"Back revolute joint angle: {obs[4]:.2f}"
    back_revolute_joint_speed = f"Back revolute joint speed: {obs[5]:.2f}"
    back_lower_leg_angle = f"Back lower leg angle: {obs[6]:.2f}"
    back_lower_leg_speed = f"Back lower leg speed: {obs[7]:.2f}"
    back_leg_ground_contact_flag = f"Back leg ground contact flag: {obs[8]:.2f}"
    front_revolute_joint_angle = f"Front revolute joint angle: {obs[9]:.2f}"
    front_revolute_joint_speed = f"Front revolute joint speed: {obs[10]:.2f}"
    front_lower_leg_angle = f"Front lower leg angle: {obs[11]:.2f}"
    front_lower_leg_speed = f"Front lower leg speed: {obs[12]:.2f}"
    front_leg_ground_contact_flag = f"Front leg ground contact flag: {obs[13]:.2f}"
    
    # Lidars
    lidar_angles = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35] # In radians starting from the top of the hull
    lidar_distances = obs[14:24]
    lidar_readings = [f"Lidar {i+1} ({angle:.2f} rad): {distance:.2f}" for i, (angle, distance) in enumerate(zip(lidar_angles, lidar_distances))]
    
    return "\n".join([starter, hull_angle_speed, angular_velocity, x_velocity, y_velocity, back_revolute_joint_angle, back_revolute_joint_speed, back_lower_leg_angle, back_lower_leg_speed, back_leg_ground_contact_flag, front_revolute_joint_angle, front_revolute_joint_speed, front_lower_leg_angle, front_lower_leg_speed, front_leg_ground_contact_flag] + lidar_readings)


####################################################################################################
# The following functions are for enhancing the observations
def walker_tilting_forward(obs):
    # Get the lidar readings
    lidar_distances = obs[14:24]
    weights = (0.5, 0.4, 0.3, 0.2, 0.1)
    THRESHOLDS = {
        "Stable": 0.7,      # Safely upright
        "Warning": 0.5,     # Approaching forward tilt
        "Critical": 0.3     # Tilting forward too much
    }
    
    WALKER_TILT = {
    "Stable": "Walker is currently stable and not tilting forward",
    "Warning": "Walker is tilting slightly forward, consider trying to balance it back",
    "Critical": "Walker is tilting forward too much, try to balance it back before it falls over"
    }
    
    
    # Calculate the weighted sum for forward and backward tilt
    forward_sum = sum(lidar_distances[5 + i] * weights[i] for i in range(5))
    backward_sum = sum(lidar_distances[i] * weights[i] for i in range(5))
    
    # Compute the tilt direction score
    weighted_sum = forward_sum - backward_sum
    
    # Determine tilt state
    if weighted_sum >= THRESHOLDS["Stable"]:
        tilt_state = "Stable"
    elif THRESHOLDS["Warning"] <= weighted_sum < THRESHOLDS["Stable"]:
        tilt_state = "Warning"
    elif weighted_sum < THRESHOLDS["Critical"]:
        tilt_state = "Critical"
    else:
        tilt_state = "Warning" 
        
    
        
    return WALKER_TILT[tilt_state]
    
def leg_movement(observations, no_check, threshold, previous_movements_back=[], previous_movements_front=[]):
    """
    Check if the legs are moving
    
    Args: 
        observations: The observations from the environment
        no_check: The number of previous movements to check
        previous_movements_back: The previous movements of the back leg
        previous_movements_front: The previous movements of the front leg
        
    Returns:
        A tuple of strings containing messages to LLM to help the agent move the legs
    """
    # Get the back leg movement
    back_movement = observations[4:8]
    front_movement = observations[9:13]
    
    previous_movements_back.append(back_movement)
    previous_movements_front.append(front_movement)
    
    if len(previous_movements_back) < no_check:
        return None, None, previous_movements_back, previous_movements_front
    
    # Get the last no_check movements
    back_movements = previous_movements_back[-no_check:]
    front_movements = previous_movements_front[-no_check:]
    
    # Check to see if the back leg is moving (Not move if the value is very close to 0)
    print(back_movement, front_movement)
    
    back_movement = (sum(back_movements) / no_check) > threshold
    front_movement = (sum(front_movements) / no_check) > threshold
    print(back_movement, front_movement)
    back_res = f"Back Leg has not moved the past {no_check} steps please try a different action to move the back leg" if not back_movement else None
    front_res = f"Front Leg has not moved the past {no_check} steps please try a different action to move the front leg" if not front_movement else None
    
    return back_res, front_res, previous_movements_back, previous_movements_front
    
def walker_moving_direction(observations):
    horizontal_velocity = observations[2]
    if horizontal_velocity < 0:
        return "Walker has leftward velocity and is moving backwards, try a different action to move forward"
    elif horizontal_velocity > 0:
        return "Walker has rightward velocity and is moving forwards, keep it up!"
    else:
        return "Walker is stationary, try a different action to move"