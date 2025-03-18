import carla
import numpy as np
import torch
import torch.nn.functional as F
import random
import time
import math
import os
import gc
import cv2

from .models import ActorCriticNet
from .replay_buffer import PrioritizedReplayBuffer
from .planning import astar_search
from .sensors import (
    attach_lidar_sensor,
    attach_camera_sensor,
    attach_collision_sensor
)
from .utils import (
    detect_red_light_in_camera,
    save_debug_image,
    save_debug_lidar
)


class A3CAgent:
    """
    RL agent using an Actor-Critic network with prioritized experience replay.
    Processes sensor data, computes rewards, updates its network, and interacts with the CARLA environment.
    """
    def __init__(self, vehicle, world, start_grid=(0,0), goal_grid=(19,19), destination_location=None, num_actions=6):
        # Environment & destination initialization.
        self.vehicle = vehicle
        self.world = world
        self.start_grid = start_grid
        self.goal_grid = goal_grid
        if destination_location is None:
            loc = self.vehicle.get_transform().location
            self.destination_location = carla.Location(x=loc.x+50, y=loc.y+50, z=loc.z)
        else:
            self.destination_location = destination_location

        # Device configuration.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        # Actor-Critic network and optimizer.
        self.actor_critic = ActorCriticNet(num_actions=self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=0.0001)

        # Rollout and replay buffer parameters.
        self.T_max = 20  # rollout length
        self.rollout = []  # store transitions (state, action, reward, log_prob, value)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        self.per_batch_size = 256

        # RL hyperparameters.
        self.gamma = 0.9
        self.entropy_coef = 0.05
        self.value_loss_coef = 0.5
        self.epsilon = 1.0
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.9999

        # Global planning grid for A* (originally for obstacle avoidance).
        self.grid_map = np.zeros((20,20), dtype=int)
        self.global_path = astar_search(self.grid_map, self.start_grid, self.goal_grid)

        # LiDAR sensor
        self.latest_lidar_data = None
        self.lidar_sensor = attach_lidar_sensor(world, vehicle)
        self.lidar_sensor.listen(self.lidar_callback)

        # Camera sensor.
        self.latest_camera_image = None
        self.camera_sensor = attach_camera_sensor(world, vehicle)
        self.camera_sensor.listen(self.camera_callback)
        self.camera_frame_count = 0

        # Collision sensor.
        self.collision_flag = False
        self.collision_penalty = 0.0
        self.collision_sensor = attach_collision_sensor(world, vehicle, self.collision_callback)
        self.episode_collision_count = 0

        # Spectator for visualization.
        self.spectator = self.world.get_spectator()

        # Tracking variables for rewards.
        self.last_movement_time = time.time()
        self.last_distance = self.vehicle.get_location().distance(self.destination_location)
        self.last_actions = []
        self.reverse_time = 0.0
        self.last_reset_episode = 0
        
        # Destination points (for toggling)
        self.destination_a = None
        self.destination_b = None

    # Sensor callbacks
    def lidar_callback(self, data):
        """
        Process incoming LiDAR data.
        """
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        self.latest_lidar_data = np.reshape(points, (-1,4))

    def camera_callback(self, image):
        """
        Process incoming camera data and update the latest camera image.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        self.latest_camera_image = array[:,:,:3]

    def collision_callback(self, event):
        """
        Process collision events; update collision penalty and count.
        """
        other = event.other_actor
        if other is not None:
            type_id = other.type_id
            if type_id.startswith("vehicle"):
                self.collision_penalty = 20
            elif type_id.startswith("walker"):
                self.collision_penalty = 40
            elif type_id.startswith("static"):
                self.collision_penalty = 12
            else:
                self.collision_penalty = 5
        else:
            self.collision_penalty = 5
        self.collision_flag = True
        self.episode_collision_count += 1

    # Processing
    def process_camera_image(self):
        """
        Convert the latest camera image into a normalized PyTorch tensor.
        
        Returns:
            img_tensor: Tensor with shape [1, C, H, W] and values normalized between 0 and 1.
        """
        if self.latest_camera_image is None:
            image = np.zeros((256,384,3), dtype=np.uint8)
        else:
            image = self.latest_camera_image
        img_tensor = torch.tensor(image.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(self.device)
        img_tensor = img_tensor / 255.0
        return img_tensor

    def plan_route(self):
        """
        Compute a global path using A* search on a grid.
        Original plan: Use LiDAR data and positions of vehicles/pedestrians to form obstacles.
        
        Returns:
            global_path: A list of grid coordinates representing the path, or None.
        """
        if self.latest_lidar_data is not None:
            self.grid_map = np.zeros((20,20), dtype=int)
            # Mark obstacles from LiDAR.
            for point in self.latest_lidar_data:
                x, y = point[0], point[1]
                # Convert to grid coordinates.
                grid_x = int((x + 100) / 10)
                grid_y = int((y + 100) / 10)
                if 0 <= grid_x < 20 and 0 <= grid_y < 20:
                    self.grid_map[grid_y, grid_x] = 1
            
            # Get vehicle location in grid coordinates.
            vehicle_loc = self.vehicle.get_location()
            start_grid_x = int((vehicle_loc.x + 100) / 10)
            start_grid_y = int((vehicle_loc.y + 100) / 10)
            start_grid = (min(max(0, start_grid_y), 19), min(max(0, start_grid_x), 19))
            
            # Get destination in grid coordinates.
            dest_grid_x = int((self.destination_location.x + 100) / 10)
            dest_grid_y = int((self.destination_location.y + 100) / 10)
            goal_grid = (min(max(0, dest_grid_y), 19), min(max(0, dest_grid_x), 19))
            
            # Compute path.
            return astar_search(self.grid_map, start_grid, goal_grid)
        return None

    def get_action(self, state):
        """
        Compute the action to take based on the current state.
        Uses epsilon-greedy exploration.
        
        Parameters:
            state: Current state tensor.
            
        Returns:
            action: Selected action index.
            log_prob: Log probability of the selected action.
            value: Estimated state value.
        """
        with torch.no_grad():
            policy_logits, value = self.actor_critic(state)
            policy = F.softmax(policy_logits, dim=1)
            
            # Epsilon-greedy exploration.
            if random.random() < self.epsilon:
                action = random.randint(0, self.num_actions - 1)
            else:
                action = policy.max(1)[1].item()
            
            log_prob = F.log_softmax(policy_logits, dim=1)[0, action]
            
        return action, log_prob, value

    def apply_control(self, action):
        """
        Apply the selected action as a control command to the vehicle.
        
        Parameters:
            action: Action index.
        """
        # Store action for cycle detection.
        self.last_actions.append(action)
        if len(self.last_actions) > 10:
            self.last_actions.pop(0)
        
        # Define control mappings.
        throttle, steer, brake, reverse = 0.0, 0.0, 0.0, False
        
        if action == 0:  # Forward
            throttle = 0.6
            steer = 0.0
            brake = 0.0
            reverse = False
        elif action == 1:  # Forward + Left
            throttle = 0.5
            steer = -0.5
            brake = 0.0
            reverse = False
        elif action == 2:  # Forward + Right
            throttle = 0.5
            steer = 0.5
            brake = 0.0
            reverse = False
        elif action == 3:  # Brake
            throttle = 0.0
            steer = 0.0
            brake = 0.8
            reverse = False
        elif action == 4:  # Reverse
            throttle = 0.5
            steer = 0.0
            brake = 0.0
            reverse = True
            self.reverse_time += 0.1
        elif action == 5:  # No operation
            throttle = 0.0
            steer = 0.0
            brake = 0.0
            reverse = False
        
        # Apply control.
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=reverse
        )
        self.vehicle.apply_control(control)

    def compute_reward(self):
        """
        Compute the reward based on the current state of the environment.
        
        Returns:
            reward: Computed reward value.
        """
        # Get current vehicle state.
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)  # km/h
        
        # Initialize reward components.
        reward = 0.0
        
        # Speed reward/penalty.
        if speed < 1.0:  # Almost stationary.
            reward -= 0.5  # Small penalty for not moving.
            current_time = time.time()
            if current_time - self.last_movement_time > 5.0:
                reward -= 5.0  # Larger penalty for being stuck.
        elif speed > 60.0:  # Too fast.
            reward -= (speed - 60.0) * 0.1  # Penalty for speeding.
        else:
            reward += 0.3  # Small reward for moving at reasonable speed.
            self.last_movement_time = time.time()
        
        # Distance to destination.
        current_distance = vehicle_location.distance(self.destination_location)
        distance_change = self.last_distance - current_distance
        
        # Reward for getting closer to destination.
        if distance_change > 0:
            reward += distance_change * 2.0
        else:
            reward += distance_change * 1.0  # Smaller penalty for moving away.
        
        self.last_distance = current_distance
        
        # Destination reached reward.
        if current_distance < 5.0:
            reward += 100.0
        
        # Collision penalty.
        if self.collision_flag:
            reward -= self.collision_penalty
            self.collision_flag = False
            self.collision_penalty = 0.0
        
        # Reverse time penalty.
        if self.reverse_time > 3.0:
            reward -= 2.0
        
        # Red light penalty.
        if detect_red_light_in_camera(self.latest_camera_image) and speed > 5.0:
            reward -= 5.0
        
        # Lane and heading penalties.
        waypoint = self.world.get_map().get_waypoint(vehicle_location)
        if waypoint:
            lane_width = waypoint.lane_width
            lane_center = waypoint.transform.location
            lane_distance = vehicle_location.distance(lane_center)
            
            # Lane deviation penalty.
            if lane_distance > lane_width / 2:
                reward -= 2.0
            
            # Heading deviation penalty.
            vehicle_forward = vehicle_transform.get_forward_vector()
            waypoint_forward = waypoint.transform.get_forward_vector()
            heading_dot = vehicle_forward.x * waypoint_forward.x + vehicle_forward.y * waypoint_forward.y
            if heading_dot < 0.7:  # More than ~45 degrees off.
                reward -= 3.0
        
        # Cycle detection (oscillatory behavior).
        if len(self.last_actions) >= 10:
            action_set = set(self.last_actions)
            if len(action_set) <= 2 and len(self.last_actions) >= 8:
                reward -= 1.0  # Penalty for repeating the same 1-2 actions.
        
        return reward

    def update_spectator(self):
        """
        Update the spectator camera to follow the vehicle.
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        spectator_transform = carla.Transform(
            vehicle_location + carla.Location(z=5.0) + vehicle_transform.get_forward_vector() * 5.0,
            carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw)
        )
        self.spectator.set_transform(spectator_transform)

    def run_rollout(self):
        """
        Run a rollout of up to T_max steps, collecting transitions.
        
        Returns:
            rollout: List of (state, action, reward, log_prob, value) tuples.
            destination_reached: Boolean indicating if destination was reached.
        """
        self.rollout = []
        destination_reached = False
        
        for _ in range(self.T_max):
            # Update spectator for visualization.
            self.update_spectator()
            
            # Get current state.
            state = self.process_camera_image()
            
            # Select action.
            action, log_prob, value = self.get_action(state)
            
            # Apply action.
            self.apply_control(action)
            
            # Wait for physics step.
            time.sleep(0.1)
            
            # Compute reward.
            reward = self.compute_reward()
            
            # Store transition.
            self.rollout.append((state, action, reward, log_prob, value))
            
            # Check if destination reached.
            current_distance = self.vehicle.get_location().distance(self.destination_location)
            if current_distance < 5.0:
                destination_reached = True
                break
        
        # Compute returns for each step in the rollout.
        returns = self.compute_returns()
        
        # Add returns to rollout.
        for i in range(len(self.rollout)):
            state, action, reward, log_prob, value = self.rollout[i]
            self.rollout[i] = (state, action, reward, log_prob, value, returns[i])
            
            # Add to replay buffer with priority = |TD error|.
            td_error = abs(returns[i] - value.item())
            self.replay_buffer.add(self.rollout[i], td_error)
        
        return self.rollout, destination_reached

    def compute_returns(self):
        """
        Compute returns for each step in the rollout using multi-step bootstrapping.
        
        Returns:
            returns: List of return values.
        """
        returns = []
        R = 0
        
        # If rollout ended early, bootstrap with value estimate.
        if len(self.rollout) < self.T_max:
            last_state = self.process_camera_image()
            with torch.no_grad():
                _, last_value = self.actor_critic(last_state)
                R = last_value.item()
        
        # Compute returns backwards.
        for i in reversed(range(len(self.rollout))):
            _, _, reward, _, _ = self.rollout[i]
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns

    def update_global(self, rollout):
        """
        Update the global network using the collected rollout.
        
        Parameters:
            rollout: List of (state, action, reward, log_prob, value, return) tuples.
            
        Returns:
            loss: Total loss value.
        """
        self.optimizer.zero_grad()
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for state, action, _, _, value, R in rollout:
            advantage = R - value.item()
            
            # Get updated policy and value.
            policy_logits, value_pred = self.actor_critic(state)
            
            # Policy loss.
            log_probs = F.log_softmax(policy_logits, dim=1)
            action_log_prob = log_probs[0, action]
            policy_loss = -action_log_prob * advantage
            policy_losses.append(policy_loss)
            
            # Value loss.
            value_loss = F.mse_loss(value_pred, torch.tensor([[R]]).to(self.device))
            value_losses.append(value_loss)
            
            # Entropy loss (for exploration).
            probs = F.softmax(policy_logits, dim=1)
            entropy = -(probs * log_probs).sum(1)
            entropy_loss = -entropy
            entropy_losses.append(entropy_loss)
        
        # Compute total loss.
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backpropagate and update.
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_from_replay(self):
        """
        Update the network using samples from the replay buffer.
        """
        if len(self.replay_buffer) < self.per_batch_size:
            return
        
        samples, indices, probs = self.replay_buffer.sample(self.per_batch_size)
        
        if not samples:
            return
        
        self.optimizer.zero_grad()
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        td_errors = []
        
        for i, (state, action, _, _, _, R) in enumerate(samples):
            # Get updated policy and value.
            policy_logits, value_pred = self.actor_critic(state)
            
            # TD error.
            td_error = R - value_pred.item()
            td_errors.append(td_error)
            
            # Policy loss.
            log_probs = F.log_softmax(policy_logits, dim=1)
            action_log_prob = log_probs[0, action]
            policy_loss = -action_log_prob * td_error
            policy_losses.append(policy_loss)
            
            # Value loss.
            value_loss = F.mse_loss(value_pred, torch.tensor([[R]]).to(self.device))
            value_losses.append(value_loss)
            
            # Entropy loss.
            probs = F.softmax(policy_logits, dim=1)
            entropy = -(probs * log_probs).sum(1)
            entropy_loss = -entropy
            entropy_losses.append(entropy_loss)
        
        # Compute total loss.
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backpropagate and update.
        loss.backward()
        self.optimizer.step()
        
        # Update priorities in replay buffer.
        self.replay_buffer.update_priorities(indices, td_errors)

    def reset(self):
        """
        Reset the agent to a new random location and toggle destination.
        """
        # Get a random spawn point.
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # Teleport vehicle.
        self.vehicle.set_transform(spawn_point)
        
        # Toggle between destinations A and B.
        if self.destination_a is not None and self.destination_b is not None:
            current_dest = self.destination_location
            if current_dest.distance(self.destination_a) < 5.0:
                self.destination_location = self.destination_b
            else:
                self.destination_location = self.destination_a
        
        # Reset tracking variables.
        self.last_distance = self.vehicle.get_location().distance(self.destination_location)
        self.last_movement_time = time.time()
        self.last_actions = []
        self.reverse_time = 0.0
        self.collision_flag = False
        self.collision_penalty = 0.0
        
        print(f"Environment reset. New spawn at: {spawn_point.location} New destination: {self.destination_location}")

    def save_model(self, filename="actor_critic_weights.pth"):
        """
        Save the model and optimizer state to a file.
        
        Parameters:
            filename: Path to save the model.
        """
        torch.save({
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="actor_critic_weights.pth"):
        """
        Load the model and optimizer state from a file.
        
        Parameters:
            filename: Path to the model file.
        """
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            print(f"Model loaded from {filename}")
        else:
            print(f"No checkpoint found at {filename}")

    def debug_camera(self, save_path="debug_camera.jpg"):
        """
        Save the current camera image for debugging.
        
        Parameters:
            save_path: Path to save the image.
        """
        save_debug_image(self.latest_camera_image, save_path)

    def debug_lidar(self, save_path="debug_lidar.png"):
        """
        Save a visualization of the current LiDAR data for debugging.
        
        Parameters:
            save_path: Path to save the visualization.
        """
        save_debug_lidar(self.latest_lidar_data, save_path) 