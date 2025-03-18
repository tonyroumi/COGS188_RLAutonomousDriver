import carla
import random
import time


def setup_carla_client(host='localhost', port=2000, timeout=10.0):
    """
    Create and return a CARLA client and world.
    
    Returns:
        client: Carla client object.
        world: Carla world object.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


def spawn_vehicle(world):
    """
    Spawn a random vehicle in the given world.
    
    Parameters:
        world: Carla world object.
        
    Returns:
        vehicle: The spawned vehicle actor.
    """
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Vehicle spawned at:", spawn_point.location)
    return vehicle


def spawn_npc_vehicles(client, world, num_vehicles=10):
    """
    Spawn a specified number of NPC vehicles and set them to autopilot.
    
    Parameters:
        client: Carla client.
        world: Carla world object.
        num_vehicles: Number of NPC vehicles to spawn.
        
    Returns:
        npc_vehicles: List of spawned NPC vehicle actors.
    """
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    npc_vehicles = []
    
    for i in range(min(num_vehicles, len(spawn_points))):
        npc_bp = random.choice(blueprint_library.filter('vehicle.*'))
        npc_vehicle = world.try_spawn_actor(npc_bp, spawn_points[i])
        if npc_vehicle is not None:
            npc_vehicles.append(npc_vehicle)
    
    traffic_manager = client.get_trafficmanager(8000)
    for npc in npc_vehicles:
        npc.set_autopilot(True, traffic_manager.get_port())
    
    return npc_vehicles


def spawn_pedestrians(world, num_pedestrians=20):
    """
    Spawn a specified number of pedestrian actors.
    
    Parameters:
        world: Carla world object.
        num_pedestrians: Number of pedestrians to spawn.
        
    Returns:
        pedestrians: List of spawned pedestrian actors.
    """
    blueprint_library = world.get_blueprint_library()
    walker_blueprints = blueprint_library.filter("walker.pedestrian.*")
    pedestrians = []
    
    for _ in range(num_pedestrians):
        bp = random.choice(walker_blueprints)
        spawn_point = world.get_random_location_from_navigation()
        if spawn_point is not None:
            try:
                ped = world.try_spawn_actor(bp, carla.Transform(spawn_point))
                if ped is not None:
                    pedestrians.append(ped)
            except Exception as e:
                print("Error spawning pedestrian:", e)
    
    return pedestrians


def is_carla_running(host="localhost", port=2000, timeout=2):
    """
    Check if CARLA server is running on the specified port.
    
    Parameters:
        host: Hostname or IP address.
        port: Port number.
        timeout: Connection timeout in seconds.
        
    Returns:
        bool: True if CARLA is running, False otherwise.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        return result == 0 