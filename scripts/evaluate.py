#!/usr/bin/env python3

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CARLA RL agent")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--num-vehicles", type=int, default=10, help="Number of NPC vehicles")
    parser.add_argument("--num-pedestrians", type=int, default=20, help="Number of pedestrians")
    parser.add_argument("--record", action="store_true", help="Record video of evaluation")
    parser.add_argument("--output-dir", type=str, default="evaluation", help="Directory to save evaluation results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if CARLA is running
    if not is_carla_running(args.host, args.port):
        print(f"CARLA simulator is NOT running on {args.host}:{args.port}. Please start CARLA and try again.")
        return
    
    print(f"CARLA simulator is running on {args.host}:{args.port}!")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    vehicle = None
    npc_vehicles = []
    pedestrians = []
    agent = None

    # Metrics for evaluation
    collisions_per_episode = []
    success_per_episode = []
    steps_per_episode = []
    episode_rewards = []

    try:
        # Setup environment
        client, world = setup_carla_client(args.host, args.port)
        
        # Setup recording if enabled
        if args.record:
            client.start_recorder(os.path.join(args.output_dir, "recording.log"))
        
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
        
        # Load trained model
        agent.load_model(args.model_path)
        
        # Set epsilon to minimum for evaluation (less random actions)
        agent.epsilon = agent.epsilon_min
        
        print("Self-Driving Agent Initialized. Starting evaluation...")
        time.sleep(2)
        
        # Evaluation loop
        for episode in range(1, args.episodes + 1):
            print(f"\nEvaluating Episode {episode}/{args.episodes}")
            
            # Reset collision count for this episode
            agent.episode_collision_count = 0
            
            # Run episode
            rollout, dest_reached = agent.run_rollout()
            
            # Compute metrics
            total_reward = sum([r for (_,_,r,_,_,_) in rollout])
            episode_steps = len(rollout)
            collisions = agent.episode_collision_count
            success_flag = 1 if dest_reached else 0
            
            # Store metrics
            episode_rewards.append(total_reward)
            collisions_per_episode.append(collisions)
            success_per_episode.append(success_flag)
            steps_per_episode.append(episode_steps)
            
            print(f"Episode {episode} Results:")
            print(f"  Reward: {total_reward:.1f}")
            print(f"  Collisions: {collisions}")
            print(f"  Success: {'Yes' if success_flag else 'No'}")
            print(f"  Steps: {episode_steps}")

            # Reset the environment for next episode
            agent.reset()
            time.sleep(1)  # Give time for reset to complete

        # Final summary
        print("\n=== Evaluation Results ===")
        print(f"Total episodes: {args.episodes}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Average Collisions: {np.mean(collisions_per_episode):.2f}")
        print(f"Success Rate: {np.mean(success_per_episode)*100:.1f}%")
        print(f"Average Steps: {np.mean(steps_per_episode):.1f}")
        
        # Save results to file
        with open(os.path.join(args.output_dir, "evaluation_results.txt"), "w") as f:
            f.write("=== Evaluation Results ===\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Total episodes: {args.episodes}\n")
            f.write(f"Average Reward: {np.mean(episode_rewards):.2f}\n")
            f.write(f"Average Collisions: {np.mean(collisions_per_episode):.2f}\n")
            f.write(f"Success Rate: {np.mean(success_per_episode)*100:.1f}%\n")
            f.write(f"Average Steps: {np.mean(steps_per_episode):.1f}\n")
        
        # Stop recording if enabled
        if args.record:
            client.stop_recorder()
        
    except Exception as e:
        print("Error during evaluation:", e)
    finally:
        print("Cleaning up actors...")
        if args.record:
            client.stop_recorder()
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