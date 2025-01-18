import carla
import random
import time
import os

def main():
    try:
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get the world and blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Select a vehicle blueprint
        # Option 1: Select a random vehicle
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

        # Option 2: Select a specific car model
        # vehicle_bp = blueprint_library.find('vehicle.audi.tt')

        # Optionally, set vehicle attributes
        # Example: Set vehicle color to red
        # vehicle_bp.set_attribute('color', '255,0,0')
        # Example: Assign role name
        # vehicle_bp.set_attribute('role_name', 'autopilot')

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            print("No spawn points available.")
            return

        # Attempt to spawn the vehicle with retries
        max_attempts = 10
        for attempt in range(max_attempts):
            spawn_point = random.choice(spawn_points)
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                print(f'Spawned {vehicle.type_id} at {spawn_point.location}.')
                break
            except RuntimeError:
                print('Spawn point occupied, retrying...')
        else:
            print('Failed to spawn vehicle after multiple attempts.')
            return

        # Retrieve the RGB camera blueprint
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        # Set camera attributes
        camera_bp.set_attribute('image_size_x', '800')  # Width of the image
        camera_bp.set_attribute('image_size_y', '600')  # Height of the image
        camera_bp.set_attribute('fov', '90')            # Field of view

        # Define the camera's relative transform (position and rotation)
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.4),   # Position relative to the vehicle
            carla.Rotation(pitch=-15.0)           # Rotation relative to the vehicle
        )

        # Spawn the camera and attach it to the vehicle
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print('RGB camera attached to the vehicle.')

        # Create a directory to save images
        output_dir = '_out'
        os.makedirs(output_dir, exist_ok=True)

	"""
        def save_image(image):
            # Save the image to disk
            image.save_to_disk(os.path.join(output_dir, f'{image.frame:06d}.png'))

        # Start listening to the camera sensor
        camera.listen(lambda image: save_image(image))
        print('Camera is now recording images.')
	"""

        # Enable autopilot
        vehicle.set_autopilot(True)
        print('Autopilot enabled for the vehicle.')

        # Define how long the vehicle should drive autonomously
        drive_duration = 120  # in seconds
        print(f'Vehicle will drive autonomously for {drive_duration} seconds.')
        time.sleep(drive_duration)

        # Cleanup: Stop camera and destroy actors
        camera.stop()
        camera.destroy()
        print('Camera stopped and destroyed.')

        vehicle.destroy()
        print('Vehicle destroyed.')

    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()

