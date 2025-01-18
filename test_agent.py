import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from .....network import RegressionNetwork
from team_code.planner import RoutePlanner

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from omegaconf import OmegaConf
from carla_gym.utils.traffic_light import TrafficLightHandler
import carla_gym.utils.transforms as trans_utils

from team_code.nav_planner import LavRoutePlanner
from team_code.waypointer import Waypointer

import yaml


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'TestAgent'


class TestAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file, route_index=None):

		self.track = autonomous_agent.Track.SENSORS
		self.alpha = 0.3
		self.status = 0
		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = deque()

		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.device = torch.device('cuda')


		# set up you agent here and load your model
		self.model = RegressionNetwork().to(self.device)
		self.model.load_state_dict(torch.load("../../../sdacreg_op034_lr0p75en5.pth", weights_only=True))
		self.model.eval()


		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()


		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0


		self.last_steers = deque()


	def _init(self):

		# self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner = RoutePlanner(3.5, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.waypointer = None


		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):

		# set up your sensors here

		return [
			{
				'type': 'sensor.camera.rgb',
				'x': 0.0, 'y': 0.0, 'z':2.0,
				'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
				'width': 320, 'height': 240, 'fov': 90,
				'id': 'rgb_front'
			},
			{
				'type': 'sensor.other.imu',
				'x': 0.0, 'y': 0.0, 'z': 0.0,
				'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
				'sensor_tick': 0.05,
				'id': 'imu'
			},
			{
				'type': 'sensor.other.gnss',
				'x': 0.0, 'y': 0.0, 'z': 0.0,
				'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
				'sensor_tick': 0.01,
				'id': 'gps'
			},
			{
				'type': 'sensor.speedometer',
				'reading_frequency': 20,
				'id': 'speed'
			},
			{
				'type': 'sensor.lidar.ray_cast',
				'x': 0.7, 'y': -0.4, 'z': 1.60,
				'roll': 0.0, 'pitch': 0.0,'yaw': -45.0,
				'id': 'LIDAR'
			},
			{
				'type': 'sensor.other.radar',
				'x': 0.7, 'y': -0.4, 'z': 1.60,
				'roll': 0.0, 'pitch': 0.0,'yaw': -45.0,
				'fov': 90,
				'id': 'RADAR'
			},
		] 

	def tick(self, input_data):
		self.step += 1

		# get your sensor output here

		rgb = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)

		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		_gps = input_data['gps'][1]

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}
		
		pos = self._get_position(result)
		result['gps'] = pos


		if self.waypointer is None:
			self.waypointer = Waypointer(
				self._global_plan, _gps
			)

		_, _, cmd = self.waypointer.tick(_gps)

		result['command'] = cmd.value

		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):

		if not self.initialized:
			self._init()

		tick_data = self.tick(input_data)
		if self.step < 1:

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gray = cv2.cvtColor(tick_data['rgb'][None], cv2.COLOR_BGR2GRAY)

		gray = torch.tensor(tick_data['rgb'][None]).permute(0,3,1,2).float().to(self.device)
		
		# run your model here, the output should be steer, throttle, and brake
		action = self.model(gray)
		cpuAction = action.cpu().detach().tolist()

		steer, throt, brake = cpuAction[1], cpuAction[0], cpuAction[2] 

		control = carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)

		return control


	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

	@staticmethod
	def _get_spawn_points(c_map):
		all_spawn_points = c_map.get_spawn_points()

		spawn_transforms = []
		for trans in all_spawn_points:
			wp = c_map.get_waypoint(trans.location)

			if wp.is_junction:
				wp_prev = wp
				# wp_next = wp
				while wp_prev.is_junction:
					wp_prev = wp_prev.previous(1.0)[0]
				spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
				if c_map.name == 'Town03' and (wp_prev.road_id == 44):
					for _ in range(100):
						spawn_transforms.append([wp_prev.road_id, wp_prev.transform])

			else:
				spawn_transforms.append([wp.road_id, wp.transform])
				if c_map.name == 'Town03' and (wp.road_id == 44):
					for _ in range(100):
						spawn_transforms.append([wp.road_id, wp.transform])

		return spawn_transforms

	def destroy(self):

		torch.cuda.empty_cache()


