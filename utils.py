import numpy as np
import pickle
from PIL import Image, ImageDraw
import random
import torchvision
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as F
import torch
# import openmesh as om
# import trimesh
import sys
from datasets import Audio2ExpDataModule

from emoca.gdl_apps.EMOCA.utils.load import load_model
from emoca.gdl.utils.FaceDetector import FAN
from emoca.gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from emoca.gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode
import os
import shutil
from pathlib import Path
from tqdm import auto
import cv2

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
		outer = [(i, i+1) for i in range(9)] + [(9, 0)]
		mouth_connections = inner + outer
		print(mouth_connections)
		print(keypoints)
		# draw_keypoints(img, keypoints, colors='white', connectivity = mouth_connections, radius=1, width=1)
		# .to(torch.uint8)
		print(img.to(torch.uint8))
		final_image_test = F.to_pil_image(img.to(torch.uint8))
		final_image_test.save('final_image_test.jpg', quality=95)
		# res = draw_keypoints(img.to(torch.uint8), keypoints, colors='white', connectivity = mouth_connections, radius=0, width=1)
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

def move_files_around(coeff_dir='/mnt/sda/AVSpeech/dataset_v2_video', metadata_dir='/mnt/sda/AVSpeech/dataset_v2_meta'):
	faulty_vid_list = []
	faulty_vid_counter = 0
	for filename in os.listdir(coeff_dir):
		try:
			# move metadata.pkl
			shutil.move(os.path.join(coeff_dir, filename, 'metadata.pkl'), os.path.join(coeff_dir, filename, filename, 'metadata.pkl'))
			# move folder with all metadata outside
			shutil.move(os.path.join(coeff_dir, filename, filename), os.path.join(metadata_dir, filename))
		except Exception as e:
			faulty_vid_counter += 1
			faulty_vid_list.append(filename)
	print(f'number of faulty vids, that werent moved: {faulty_vid_counter}')
	print(f'faulty_vid_list: {faulty_vid_list}')


def move_files_around_nr(dataset_dir='/mnt/sda/AVSpeech/NR_v1_dataset/dataset_v1'):
	for video_name in os.listdir(dataset_dir):
		# first move metadata
		video_dir = os.path.join(dataset_dir, video_name, 'dataset_preprocessed', video_name)
		metadata_dir = os.path.join(dataset_dir, video_name, 'dataset_preprocessed', video_name, 'metadata.pkl')
		more_metadata_dir = os.path.join(dataset_dir, video_name, 'dataset_preprocessed', video_name, video_name)
		final_metadata_dir = '/mnt/sda/AVSpeech/NR_v1_dataset/dataset_v1_meta'
		shutil.move(metadata_dir, more_metadata_dir)
		shutil.move(more_metadata_dir, final_metadata_dir)


		final_dir = os.path.join(dataset_dir, video_name)
		# move all the files from there to the parent final dir
		shutil.move(video_dir, final_dir)

	
def move_files_some_more():
	dataset_dir='/mnt/sda/AVSpeech/NR_v1_dataset/dataset_v1'
	# delete empty dir, move folder one up
	for video_name in os.listdir(dataset_dir):
		os.rmdir(os.path.join(dataset_dir, video_name, 'dataset_preprocessed'))
		# move the content one up in dir
		for item in os.listdir(os.path.join(dataset_dir, video_name, video_name)):
		
			shutil.move(os.path.join(dataset_dir, video_name, video_name, item), os.path.join(dataset_dir, video_name))
		# delete empty child dir
		os.rmdir(os.path.join(dataset_dir, video_name, video_name))


def get_Om(pose, shape, exp, emoca=None, batch_size=64):
	"""
	returns 3d coordinates of Flame model, based on pose, shape, exp
	"""
	path_to_models = "root/MemFace/emoca/assets/EMOCA/models"
	model_name = 'EMOCA'
	mode = 'detail'
	
	if emoca == None:
		emoca, conf = load_model(path_to_models, model_name, mode)
		emoca.cuda()
		emoca.eval()
	
	if batch_size != 0:	
		pose = torch.reshape(pose, (pose.shape[0]*pose.shape[1], pose.shape[2]))
		shape = torch.reshape(shape, (shape.shape[0]*shape.shape[1], shape.shape[2]))
		exp = torch.reshape(exp, (exp.shape[0]*exp.shape[1], exp.shape[2]))
		print(f'pose.shape: {pose.shape}')
		print(f'shape.shape: {shape.shape}')
		print(f'exp.shape: {exp.shape}')
		verts, landmarks2d, landmarks3d_hat = emoca.deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)
		print(f'landmarks3d_hat.shape after get_Om: {landmarks3d_hat.shape}')
		sequence_length = landmarks3d_hat.shape[0] // batch_size
		landmarks3d_hat = landmarks3d_hat.view(batch_size, sequence_length, landmarks3d_hat.shape[1], landmarks3d_hat.shape[2])
		print(f'landmarks3d_hat.shape batched: {landmarks3d_hat.shape}')
		landmarks3d_hat = landmarks3d_hat[:,:,48:,:]
	
	else:
		codedict = {}
		codedict['shapecode'] = shape
		codedict['expcode'] = exp
		codedict['posecode'] = pose
		verts, landmarks2d, landmarks3d_hat = emoca.deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)
	# 	print(f'landmarks3d: {landmarks3d}')
	
	return landmarks3d_hat


def neural_rendering_facereconstruction(filepath):
	path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
	input_video =	filepath 
	output_folder = os.path.join(filepath.rsplit('.', 1)[0], 'dataset_preprocessed')
	model_name = 'EMOCA_v2_lr_mse_20'
	image_type = 'geometry_detail'
	cat_dim = 0
	include_transparent = False
	processed_subfolder = None
	mode = 'detail'
	
	dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=Path(input_video).stem,
        batch_size=1, num_workers=24)
	dm.prepare_data()
	dm.setup()
	processed_subfolder = Path(dm.output_dir).name

	## 2) Load the model
	emoca, conf = load_model(path_to_models, model_name, mode)
	emoca.cuda()
	emoca.eval()

	outfolder = str(Path(output_folder) / Path(input_video).stem)

	## 3) Get the data loadeer with the detected faces
	dl = dm.test_dataloader()
	## 4) Run the model on the data
	for j, batch in enumerate (auto.tqdm( dl)):

		current_bs = batch["image"].shape[0]
		img = batch
		vals, visdict = test(emoca, img)
		# print('+---------------- Landmarks3D ----------------+')
		# print(vals['landmarks3d'])
		for i in range(current_bs):
			# name = f"{(j*batch_size + i):05d}"
			name =  batch["image_name"][i]

			sample_output_folder = Path(outfolder) /name
			sample_output_folder.mkdir(parents=True, exist_ok=True)
			# break
			save_images(outfolder, name, visdict, i)
			save_codes(Path(outfolder), name, vals, i)


def find_most_similar_tensors(K_nr):
	# Create an empty array to store RMS distances between tensors
	print('inside find_most_similar_tensors')
	n = len(K_nr)
	print(f'len(K_nr): {n}')
	print(f'first elem of K_nr: {K_nr[0]}')
	rms_distances = np.full((n, n), np.inf)	
	# Calculate RMS distances between all pairs of tensors
	for i, (tensor1, _) in enumerate(K_nr):
		for j, (tensor2, _) in enumerate(K_nr):
			if i != j:
				rms_distances[i, j] = torch.sqrt(torch.mean(torch.square(tensor1 - tensor2)))
	
	# Find the indices of the two tuples with the smallest RMS distance
	min_indices = np.unravel_index(np.argmin(rms_distances), rms_distances.shape)
	
	# Return the two tuples with the smallest RMS distance and the RMS distance itself
	return K_nr[min_indices[0]], K_nr[min_indices[1]], rms_distances[min_indices]


def replace_tuple_by_label(array_of_tuples, target_label, new_tuple):
    result = []
    for tuple_a in array_of_tuples:
        if tuple_a[1] == target_label:
            result.append(new_tuple)
        else:
            result.append(tuple_a)
    return result

def find_optimal_lipsbox(landmarks2d_dir='/home/avocoral/MemFace/williamblake10/williamblake10_meta/landmarks'):
	minX, maxX, minY, maxY = None, None, None, None
	for frame in os.listdir(landmarks2d_dir):
		landmarks2d_path = os.path.join(landmarks2d_dir, frame)
		
		# load mouth landmarks2d
		objects = []
		with (open(landmarks2d_path, "rb")) as openfile:
			while True:
				try:
					objects.append(pickle.load(openfile))
				except EOFError:
					break
		
		mouth_landmarks = torch.Tensor([[[int(objects[1][i][0]), int(objects[1][i][1])] for i in range(len(objects[1])) if 48 <= i < 69]])
		print(f'mouth_landmarks: {mouth_landmarks}')
		
		# lip region crop + 1pixel boundary around 
		minX_new = int(mouth_landmarks[:, :, 0].min() - 1)
		maxX_new = int(mouth_landmarks[:, :, 0].max() + 1)
		minY_new = int(mouth_landmarks[:, :, 1].min() - 1)
		maxY_new = int(mouth_landmarks[:, :, 1].max() + 1)

		if minX == None or minX_new < minX:
			minX = minX_new
		if maxX == None or maxX_new > maxX:
			maxX = maxX_new
		if minY == None or minY_new < minY:
			minY = minY_new
		if maxY == None or maxY_new > maxY:
			maxY = maxY_new
	
	return minX, maxX, minY, maxY


def construct_explicitmem(data_dir='/home/avocoral/MemFace/williamblake10/williamblake10', metadata_dir='/home/avocoral/MemFace/williamblake10/williamblake10_meta'):
	
	K_nr_filepath = '/home/avocoral/Downloads/Obamaset/K_nr.pt'
	V_nr_filepath = '/home/avocoral/Downloads/Obamaset/V_nr.pt'
	N = 300

	if os.path.exists(K_nr_filepath) and os.path.exists(V_nr_filepath):
		K_nr = torch.load(K_nr_filepath)
		V_nr = torch.load(V_nr_filepath)
		return K_nr, V_nr
	
	# Step 0: Build K_all
	K_all = []
	for i, frame_name in enumerate(os.listdir(data_dir)):
		if i == 900:
			break
		K_all.append((torch.from_numpy(np.load(os.path.join(data_dir, frame_name, 'landmarks3d.npy')))[:, 48:, :], frame_name))

	# step 1: Initialize K_nr, V_nr
	K_nr = random.sample(K_all, N)

	# step 2: Find two most similar mouth shapes:
	k_m1, k_m2, Dmin = find_most_similar_tensors(K_nr)
	print(f'k_m1: {k_m1}')
	print(f'k_m2: {k_m2}')
	print(f'Dmin: {Dmin}')
	
	# step 3: go through all tensors in K_all
	i = 0
	for k_tmp in K_all:
		print(f"Preprocessed {i}/{len(K_all)}")
		# create a copy of K_nr where k_m1 replaced with k_tmp
		K_tmp1 = replace_tuple_by_label(K_nr, k_m1[1], k_tmp)	
		# find two most similar tensors in K_nr and their distance Dmin1
		# print(f'K_tmp1[0]: {K_tmp1[0]}')
		_ , _ , Dmin1 = find_most_similar_tensors(K_tmp1)
		# create a copy of K_nr where k_m1 replaced with k_tmp
		K_tmp2 = replace_tuple_by_label(K_nr, k_m2[1], k_tmp)	
		# find two most similar tensors in K_nr and their distance Dmin1
		_ , _ , Dmin2 = find_most_similar_tensors(K_tmp2)

		if max(Dmin1, Dmin2) > Dmin:
			if Dmin1 > Dmin2:
				K_nr = replace_tuple_by_label(K_nr, k_m1[1], k_tmp)
			else:
				K_nr = replace_tuple_by_label(K_nr, k_m2[1], k_tmp)
		k_m1, k_m2, Dmin = find_most_similar_tensors(K_nr)
		i += 1
	resulting_labels = [elem[1] for elem in K_nr]
	print(resulting_labels)
	
	# find the lipsbox
	# minX, maxX, minY, maxY = find_optimal_lipsbox()
	
	# extract V_nr
	# lips_dir = '/home/avocoral/MemFace/williamblake10/williamblake10_ExplicitMemLips'
	# print(f'minX, maxX, minY, maxY = {minX}, {maxX}, {minY}, {maxY}')
	# print(f'len(K_nr)')
	# for (_, frame) in K_nr:
	# 	print(f'frame: {frame}')
	# 	img_path = os.path.join(data_dir, frame, 'inputs.png')
	# 	original_image = cv2.imread(img_path)
	# 	crop_lips = original_image[minY:maxY, minX:maxX]
	# 	cv2.imwrite(f'{lips_dir}/{frame}_lips.png', crop_lips)	

	
	# return pytorch tensors of tuple tensors [[k_nr, v_nr], ...]
	# k_nr - landmarks3d, v_nr image 256x256x3
	image_path = '/home/avocoral/Downloads/Obamaset/Obama_vid/{}/inputs.png'
	K_nr_new = torch.stack([tensor.squeeze().reshape(60, -1) for tensor, image_name in K_nr]) 
	V_nr_new = torch.stack([load_tensor_image(image_path.format(image_name)) for tensor, image_name in K_nr])
	
	# save K_nr_new and V_nr_new
	torch.save(K_nr_new, K_nr_filepath)
	torch.save(V_nr_new, V_nr_filepath)
	
	return K_nr_new, V_nr_new


def load_tensor_image(image_name):
	with Image.open(image_name) as image:
		# only load the lower part of the image with lips
		image = image.crop((0, image.size[1]//2, image.size[0], image.size[1]))
		image = image.convert("RGB")
		tensor_image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
		tensor_image = tensor_image.view(image.size[1], image.size[0], -1)
		tensor_image = tensor_image.permute(2, 0, 1).float().div(255.0)
	return tensor_image

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
	# move_files_around()
	
	# --- to load the dataloader and take the 1st batch ---
	# datamodule = Audio2ExpDataModule()
	# datamodule.setup()
	# train_dataloader = datamodule.train_dataloader()	
	# first_batch = next(iter(train_dataloader))
	# 
	# packed_audio_embed, packed_exp, packed_pose, packed_shape, packed_landmarks3d, sequence_lengths = first_batch

	# audio_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_audio_embed, batch_first=True)
	# exp, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_exp, batch_first=True)
	# pose, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_pose, batch_first=True)
	# shape, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_shape, batch_first=True)
	# landmarks3d, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_landmarks3d, batch_first=True)
	# 
	# landmarks3d_hat = get_Om(pose, shape, exp)
	# print(f'landmarks_hat.shape: {landmarks3d_hat.shape}')
	
	# --- preprocess NR dataset ---
	# dataset_dir = '/mnt/sda/AVSpeech/NR_v1_dataset/dataset_v1_videos'
	# 
	# for i, video_name in enumerate(os.listdir(dataset_dir)):
	# 		print(f'! video #{i} preprocessed !')
	# 		filepath = os.path.join(dataset_dir, video_name)
	# 
	filepath = '/home/avocoral/Downloads/Obamaset/Obama_vid.mp4'
	neural_rendering_facereconstruction(filepath)

	# K_nr, V_nr = construct_explicitmem()
	# print(f'K_nr.shape: {K_nr.shape}')
	# print(f'V_nr.shape: {V_nr.shape}')
	# print(find_optimal_lipsbox())
	# move_files_around_nr()
	# move_files_some_more()
