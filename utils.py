import FaceVerse as fv
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
from gdl import 

def faceReconstruction(filepath):
	
	return None

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


def readCoeff(shape_filepath, exp_filepath, pose_filepath):
	print('+----------- Exp ------------+')
	exp = np.load(exp_filepath, allow_pickle=True)
	print(exp)
	print(f"len: {len(exp)}")

	print('+----------- Pose ------------+')
	pose = np.load(pose_filepath, allow_pickle=True)
	print(pose)
	print(f"len: {len(pose)}")
	
	print('+----------- Shape ------------+')
	shape = np.load(shape_filepath, allow_pickle=True)
	print(shape)
	print(f"len: {len(shape)}")

	return shape, exp, pose


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


def getOm(obj, landmark):
	mouth_landmarks = landmarks[]
	objects = []
	with (open(landmark_filepath, "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break
	mouth_landmarks = objects[1][48:69]
	# mouth_landmarks = torch.Tensor([[[int(objects[1][i][0]), int(objects[1][i][1])] for i in range(len(objects[1])) if 48 <= i < 69]])
	mesh = trimesh.load_mesh(obj_filepath)


	# find closest projections from mouth_landmarks to mesh verticies

	
	return None

def reconstructFlame(shapecode, expcode, posecode):
	verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode)
	print(landmarks3d)
	return landmarks3d

if __name__ == '__main__':
	landmark = '/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/landmarks/000042_000.pkl'
	exp_filepath = '/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/results/EMOCA/000042_000/exp.npy'
	pose_filepath = '/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/results/EMOCA/000042_000/pose.npy'
	shape_filepath = '/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/results/EMOCA/000042_000/shape.npy'
	obj_filepath = '/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/results/EMOCA/000042_000/mesh_coarse_detail.obj'
	# readLandmarks(landmark, only_mouth=True, visualize=True)
	shapecode, expcode, posecode = readCoeff(shape_filepath, exp_filepath, pose_filepath)

	# readObj(obj_filepath)
