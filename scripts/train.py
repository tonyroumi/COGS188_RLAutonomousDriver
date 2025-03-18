#!/usr/bin/env python3

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import gc
import torch
import os
import carla

from carla_rl_agent.environment import (
    setup_carla_client,
    spawn_vehicle,
    spawn_npc_vehicles,
    spawn_pedestrians,
    is_carla_running
)
from carla_rl_agent.agent import A3CAgent
from carla_rl_agent.utils import plot_rewards


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CARLA RL agent")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save models and plots")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load model from")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--num-vehicles", type=int, default=10, help="Number of NPC vehicles")
    parser.add_argument("--num-pedestrians", type=int, default=20, help="Number of pedestrians")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if CARLA is running
    if not is_carla_running(args.host, args.port):
        print(f"CARLA simulator is NOT running on {args.host}:{args.port}. Please start CARLA and try again.")
        return
    
    print(f"CARLA simulator is running on {args.host}:{args.port}!")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    vehicle = None
    npc_vehicles = []
    pedestrians = []
    agent = None

    # Metrics for evaluation
    collisions_per_episode = []
    success_per_episode = []
    steps_per_episode = []
    episode_rewards = []
    cumulative_reward = 0.0

    try:
        # Setup environment
        client, world = setup_carla_client(args.host, args.port)
        vehicle = spawn_vehicle(world)
        npc_vehicles = spawn_npc_vehicles(client, world, num_vehicles=args.num_vehicles)
        pedestrians = spawn_pedestrians(world, num_pedestrians=args.num_pedestrians)
        
        # Define destinations
        start_grid = (0, 0)
        goal_grid = (19, 19)
        dest_a = carla.Location(x=100, y=100, z=vehicle.get_transform().location.z)
        dest_b = carla.Location(x=-100, y=-100, z=vehicle.get_transform().location.z)

        # Initialize agent
        agent = A3CAgent(vehicle, world, start_grid, goal_grid, destination_location=dest_a, num_actions=6)
        agent.destination_a = dest_a
        agent.destination_b = dest_b
        
        # Load model if specified
        if args.load_model:
            agent.load_model(args.load_model)

        # Debug sensor data at startup
        agent.debug_camera(save_path=os.path.join(args.save_dir, "debug_camera.jpg"))
        agent.debug_lidar(save_path=os.path.join(args.save_dir, "debug_lidar.png"))
        
        print("Self-Driving Agent Initialized. Starting training...")
        time.sleep(5)
        
        # Training loop
        for episode in range(1, args.episodes + 1):
            # Reset collision count for this episode
            agent.episode_collision_count = 0
            
            # Run episode
            rollout, dest_reached = agent.run_rollout()
            loss = agent.update_global(rollout)
            
            # Additional replay buffer updates
            for _ in range(5):
                agent.update_from_replay()
            
            # Compute metrics
            total_reward = sum([r for (_,_,r,_,_,_) in rollout])
            cumulative_reward += total_reward
            episode_steps = len(rollout)
            collisions = agent.episode_collision_count
            success_flag = 1 if dest_reached else 0
            
            # Store metrics
            episode_rewards.append(total_reward)
            collisions_per_episode.append(collisions)
            success_per_episode.append(success_flag)
            steps_per_episode.append(episode_steps)
            
            # Epsilon decay
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            
            print(f"Episode {episode}: Reward = {total_reward:.1f}, "
                  f"Collisions = {collisions}, "
                  f"Success = {success_flag}, "
                  f"Steps = {episode_steps}, "
                  f"Epsilon = {agent.epsilon:.3f}")

            # Reset the environment if destination reached or 20 episodes passed since last reset
            if dest_reached or (episode - agent.last_reset_episode) >= 20:
                agent.reset()
                agent.last_reset_episode = episode

            # Periodic evaluation and saving
            if episode % 100 == 0:
                # Evaluate average metrics
                avg_reward = np.mean(episode_rewards[-100:])
                avg_collisions = np.mean(collisions_per_episode[-100:])
                success_rate = np.mean(success_per_episode[-100:])
                avg_steps = np.mean(steps_per_episode[-100:])
                
                print(f"After {episode} episodes:")
                print(f"  Avg. Reward: {avg_reward:.2f}")
                print(f"  Avg. Collisions: {avg_collisions:.2f}")
                print(f"  Success Rate: {success_rate*100:.1f}%")
                print(f"  Avg. Steps: {avg_steps:.1f}")

                # Plot reward graph
                plot_rewards(episode_rewards, 
                             os.path.join(args.save_dir, f"reward_graph_episode_{episode}.png"))

                # Save model checkpoint
                agent.save_model(os.path.join(args.save_dir, "actor_critic_weights.pth"))

                # Clear GPU cache and run garbage collection
                torch.cuda.empty_cache()
                gc.collect()

        # Final summary
        print("\n=== Final Results ===")
        print(f"Total episodes: {args.episodes}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Average Collisions: {np.mean(collisions_per_episode):.2f}")
        print(f"Success Rate: {np.mean(success_per_episode)*100:.1f}%")
        print(f"Average Steps: {np.mean(steps_per_episode):.1f}")
        
    except Exception as e:
        print("Error during setup or training:", e)
    finally:
        print("Cleaning up actors...")
        if vehicle is not None:
            vehicle.destroy()
        for npc in npc_vehicles:
            npc.destroy()
        for ped in pedestrians:
            ped.destroy()
        if agent is not None:
            if agent.lidar_sensor is not None:
                agent.lidar_sensor.destroy()
            if agent.camera_sensor is not None:
                agent.camera_sensor.destroy()
            if hasattr(agent, 'collision_sensor') and agent.collision_sensor is not None:
                agent.collision_sensor.destroy()


if __name__ == "__main__":
    main() 