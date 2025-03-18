import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def detect_red_light_in_camera(camera_image):
    """
    Use HSV thresholding on the camera image to detect a red light.
    Specifically, targeting the central region of the camera image.
    
    Parameters:
        camera_image: RGB image as numpy array.
        
    Returns:
        Boolean: True if a red light is detected, False otherwise.
    """
    if camera_image is None:
        return False
    
    hsv = cv2.cvtColor(camera_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    h, w = mask.shape
    cx, cy = w//2, h//2
    region = mask[cy - h//6: cy + h//6, cx - w//6: cx + w//6]
    red_pixels = cv2.countNonZero(region)
    total_pixels = region.size
    
    return (red_pixels/total_pixels) > 0.2


def save_debug_image(image, save_path="debug_camera.jpg"):
    """
    Save an image for debugging purposes.
    
    Parameters:
        image: Image as numpy array.
        save_path: Path to save the image.
    """
    if image is not None:
        cv2.imwrite(save_path, image)
        print(f"Saved image to {save_path}")
    else:
        print("No image available to save.")


def save_debug_lidar(lidar_data, save_path="debug_lidar.png"):
    """
    Save a scatter plot of LiDAR data for debugging.
    
    Parameters:
        lidar_data: LiDAR points as numpy array.
        save_path: Path to save the plot.
    """
    if lidar_data is not None:
        plt.figure(figsize=(6, 6))
        plt.scatter(lidar_data[:, 0], lidar_data[:, 1], s=1)
        plt.title("LiDAR X-Y Scatter")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved LiDAR scatter plot to {save_path}")
    else:
        print("No LiDAR data available to save.")


def save_model(model, optimizer, epsilon, filename="actor_critic_weights.pth"):
    """
    Save model, optimizer states, and epsilon value to a file.
    
    Parameters:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        epsilon: Exploration parameter.
        filename: Path to save the checkpoint.
    """
    torch.save({
        "actor_critic_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epsilon": epsilon
    }, filename)
    print(f"Model saved to {filename}")


def load_model(model, optimizer, filename="actor_critic_weights.pth"):
    """
    Load model and optimizer states from a checkpoint file.
    
    Parameters:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        filename: Path to the checkpoint file.
        
    Returns:
        epsilon: Loaded epsilon value.
    """
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["actor_critic_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epsilon = checkpoint["epsilon"]
        print(f"Model loaded from {filename}")
        return epsilon
    else:
        print(f"No checkpoint found at {filename}")
        return 1.0  # Default epsilon


def plot_rewards(rewards, filename="reward_plot.png"):
    """
    Plot and save a graph of rewards over episodes.
    
    Parameters:
        rewards: List of episode rewards.
        filename: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Reward vs. Episode")
    plt.savefig(filename)
    plt.close()
    print(f"Saved reward plot to {filename}") 