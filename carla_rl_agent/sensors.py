import carla
import numpy as np


def attach_lidar_sensor(world, vehicle):
    '''
    Attach a LiDAR sensor to the vehicle.
    
    Original plan: Use LiDAR data for obstacle detection and global planning.
    
    Returns:
        lidar_sensor: The spawned LiDAR sensor actor.
    '''
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('sensor_tick', '0.1')
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    return lidar_sensor


def attach_camera_sensor(world, vehicle):
    """
    Attach a camera sensor to the vehicle.
    
    Returns:
        camera_sensor: The spawned camera sensor actor.
    """
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '384')
    camera_bp.set_attribute('image_size_y', '256')
    camera_bp.set_attribute('fov', '135')
    camera_bp.set_attribute('sensor_tick', '0.1')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera_sensor


def attach_collision_sensor(world, vehicle, collision_callback):
    """
    Attach a collision sensor to the vehicle and register the callback.
    
    Parameters:
        world: Carla world object.
        vehicle: Vehicle actor.
        collision_callback: Function to handle collision events.
        
    Returns:
        collision_sensor: The spawned collision sensor actor.
    """
    blueprint_library = world.get_blueprint_library()
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_sensor.listen(collision_callback)
    return collision_sensor


def process_camera_image(image_data):
    """
    Process raw camera data into a numpy array.
    
    Parameters:
        image_data: Raw CARLA camera data.
        
    Returns:
        numpy.ndarray: Processed image as a numpy array.
    """
    array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image_data.height, image_data.width, 4))
    return array[:, :, :3]  # Remove alpha channel


def process_lidar_data(lidar_data):
    """
    Process raw LiDAR data into a numpy array.
    
    Parameters:
        lidar_data: Raw CARLA LiDAR data.
        
    Returns:
        numpy.ndarray: Processed LiDAR points as a numpy array.
    """
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    return np.reshape(points, (-1, 4))  # x, y, z, intensity 