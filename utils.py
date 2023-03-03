import numpy as np
import pickle
from PIL import Image, ImageDraw
import random
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_keypoints, save_image
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as F
import torch
import openmesh as om
import trimesh
import sys
sys.path.insert(0, '/home/avocoral/MemFace/emoca')

from gdl_apps.EMOCA.utils.load import load_model
from gdl.utils.FaceDetector import FAN
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode
import os
import shutil

def readReconstruction(filepath):
	f = np.load(filepath, allow_pickle=True)
	print(f)
	print(f.shape())
	for i in f:
		print(f"i:{i}")
		print(f"len of i:{len(i)}")


def readLandmarks(landmark_filepath, only_mouth = False, visualize = False):
	objects = []
	with (open(landmark_filepath, "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break
	if only_mouth == True:
		# mouth_landmarks = objects[1][48:69]
		mouth_landmarks = torch.Tensor([[[int(objects[1][i][0]), int(objects[1][i][1])] for i in range(len(objects[1])) if 48 <= i < 69]])
		inner_mouth_landmarks = [[objects[1][i][0], objects[1][i][1]] for i in range(len(objects[1])) if 60 <= i < 69]
		inner_mouth_landmarks.append(inner_mouth_landmarks[0])
		outer_mouth_landmarks = [[objects[1][i][0], objects[1][i][1]] for i in range(len(objects[1])) if 48 <= i < 60]
		outer_mouth_landmarks.append(outer_mouth_landmarks[0])
		print(mouth_landmarks)
		print(len(mouth_landmarks))
		return mouth_landmarks

	if visualize == True:
		# create black canvas
		vis_name = f'vis_{random.randint(100000,999999)}.jpg'
		# im = Image.new('RGB', (256, 256), (0, 0, 0))
		original_image = Image.open(r'/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/videos/000042.png')

		original_image.save('orig_test.jpg', quality=95)
		img = torchvision.transforms.functional.to_tensor(original_image).cuda()
		# im.save(vis_name, quality=95)
		# draw mouth landmarks
		# img = pil_to_tensor(im)
		# img = read_image(vis_name)
		# keypoints = torch.Tensor([mouth_landmarks])
		keypoints = mouth_landmarks
		inner = [(i, i+1) for i in range(10, 18)] + [(18, 10)]
		outer = [(i, i+1) for i in range(9)] + [(9, 0)]
		mouth_connections = inner + outer
		print(mouth_connections)
		print(keypoints)
		# draw_keypoints(img, keypoints, colors='white', connectivity = mouth_connections, radius=1, width=1)
		# .to(torch.uint8)
		print(img.to(torch.uint8))
		final_image_test = F.to_pil_image(img.to(torch.uint8))
		final_image_test.save('final_image_test.jpg', quality=95)
		res = draw_keypoints(img.to(torch.uint8), keypoints, colors='white', connectivity = mouth_connections, radius=0, width=1)
		vis_name = f'vis_{random.randint(100000,999999)}.jpg'
		final_image = F.to_pil_image(res)
		# save_image(final_image, vis_name)
		final_image.save(vis_name, quality=95)
		print(f'visualization filename: {vis_name}')


def readCoeff(shape_filepath, exp_filepath, pose_filepath, cam_filepath):
	print('+----------- Exp ------------+')
	exp = np.load(exp_filepath, allow_pickle=True)
	print(torch.from_numpy(exp))
	print(f"len: {len(exp)}")

	print('+----------- Pose ------------+')
	pose = np.load(pose_filepath, allow_pickle=True)
	print(torch.from_numpy(pose))
	print(f"len: {len(pose)}")
	
	print('+----------- Shape ------------+')
	shape = np.load(shape_filepath, allow_pickle=True)
	print(shape)
	print(f"len: {len(shape)}")

	print('+----------- Cam ------------+')
	cam = np.load(cam_filepath, allow_pickle=True)
	print(cam)
	print(f"len: {len(cam)}")

	return shape, exp, pose, cam


def readObj(obj_filepath):
	# mesh = om.read_trimesh(obj_filepath)
	# with open(obj_filepath) as f:
	# 	lines = f.readlines()
	# vertices = [line for line in lines if line.startswith('v ')]
	# faces = [line for line in lines if line.startswith('f ')]
	mesh = trimesh.load_mesh(obj_filepath)	
	print('+----------- Mesh ------------+')
	print(mesh.vertices)
	print(f'len: {mesh.vertices.shape}')
	return None

def move_files_around(coeff_dir='/mnt/sda/AVSpeech/video', metadata_dir='/mnt/sda/AVSpeech/metadata'):
	for filename in os.listdir(coeff_dir):
		# move metadata.pkl
		shutil.move(os.path.join(coeff_dir, filename, 'metadata.pkl'), os.path.join(coeff_dir, filename, filename, 'metadata.pkl'))
		# move folder with all metadata outside
		shutil.move(os.path.join(coeff_dir, filename, filename), os.path.join(metadata_dir, filename))

def get_Om(pose, shape, exp):
	"""
	returns 3d coordinates of Flame model, based on pose, shape, exp
	"""
	path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
	model_name = 'EMOCA'
	mode = 'detail'
	emoca, conf = load_model(path_to_models, model_name, mode)
	emoca.cuda()
	emoca.eval()
		
	codedict = {}
	codedict['shapecode'] = shape
	codedict['expcode'] = exp
	codedict['posecode'] = pose
	verts, landmarks2d, landmarks3d = emoca.deca.flame(shape_params=torch.from_numpy(shape).unsqueeze(0).to('cuda:0'), expression_params=torch.from_numpy(exp).unsqueeze(0).to('cuda:0'),pose_params=torch.from_numpy(pose).unsqueeze(0).to('cuda:0'))

	print(f'landmarks3d: {landmarks3d}')
	return landmarks3d


if __name__ == '__main__':
	# landmark = '/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/landmarks/000042_000.pkl'
	# exp_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/exp.npy'
	# pose_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/pose.npy'
	# shape_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/shape.npy'
	# cam_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/cam.npy'
	# shapecode, expcode, posecode, camcode = readCoeff(shape_filepath, exp_filepath, pose_filepath, cam_filepath)
	# print(f'dim of shape: {torch.from_numpy(shapecode).ndimension()}')
	# print(f'dim of exp: {torch.from_numpy(expcode).ndimension()}')
	# betas = torch.cat([torch.from_numpy(shapecode).unsqueeze(0), torch.from_numpy(expcode).unsqueeze(0)], dim=1)
	# print(f'betas: {betas}')
	# landmarks3d = get_Om(posecode, shapecode, expcode)
	# print(landmarks3d)
	move_files_around()
