import gymnasium as gym

def obs_to_text_experiment_1(obs):
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
    