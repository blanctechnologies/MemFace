from neuralrendering import *
from neuralrendering import NeuralRender
from datasets import NeuralRenderingDataset
import os
import torch
from PIL import Image
from moviepy.editor import VideoFileClip
import imageio

from pathlib import Path
from emoca.gdl_apps.EMOCA.utils.io import decode
from gdl.utils.lightning_logging import _fix_image
from skimage.io import imsave
from emoca.gdl.datasets.ImageTestDataset import TestData
from emoca.gdl_apps.EMOCA.utils.load import load_model
from emoca.gdl.utils.FaceDetector import FAN
from emoca.gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from emoca.gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test, decode

from audio2exp_inference import *
from audio2exp_inference import frames_to_video
import numpy as np
from utils import get_Om
from audio2exp_inference import extractAudioEncoding
from audio2exp_inference import audio2expression
from utils import create_reconstruction_from_vals

def load_NR_model():
	device = 'cuda:0'
	NeuralRender = NeuralRender()
	checkpoints = torch.load('/home/avocoral/MemFace/checkpoints/MemFace/243pt0fh/checkpoints/21287.ckpt')
	NeuralRender.load_state_dict(checkpoints["state_dict"])
	NeuralRender.to(device)
	NeuralRender.eval()
	print('NR model loaded!')
	return NeuralRender

# prep the data first 15 seconds of masked_ref_images, expressions from last 15 seconds
# 1. get masked images
def get_masked_imgs():
	masked_images = []
	NRdataset = NeuralRenderingDataset()
	for i in range(10):
		masked_ref_image, _, _  = NRdataset.__getitem__(i*30)
		masked_images.append(masked_ref_image[:, :3, :, :])
		
	masked_images = torch.stack(masked_images).view(10, 30, 3, 224, 224).to('cuda')
	# save dataset ref images
	return masked_images


# 2. get ref images
# take old pose and shape and create tensor [new exp, old pose, old shape]
def create_ref_images():
	coeff_dir ='/home/avocoral/Downloads/Obamaset/Obama_vid/Obama'
	frame_num = len(os.listdir(coeff_dir))
	frame_names = [str(i).zfill(6)+'_000' for i in range(1,301)]
	print(f'total num of inference frames: {len(frame_names)}')
	print(f'first frame name: {frame_names[0]}')
	print(f'last frame name: {frame_names[-1]}')


	old_pose = []
	old_shape = []
	old_tex = []
	old_cam = []
	old_detail = []
	for i, frame_name in enumerate(frame_names):
		old_pose.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'pose.npy'))))
		old_shape.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'shape.npy'))))
		old_tex.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'tex.npy'))))
		old_cam.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'cam.npy'))))
		old_detail.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'detail.npy'))))

	old_pose = torch.stack(old_pose)
	old_shape = torch.stack(old_shape)
	old_tex = torch.stack(old_tex)
	old_cam = torch.stack(old_cam)
	old_detail = torch.stack(old_detail)

	new_exp = []
	for i, frame_name in enumerate([str(i).zfill(6)+'_000' for i in range(301, 601)]):
		new_exp.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'exp.npy'))))
	new_exp = torch.stack(new_exp)

	print(f'old params and new exp are loaded!')

	path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
	model_name = 'EMOCA_v2_lr_mse_20'
	mode = 'detail'
	final_out_folder = '/home/avocoral/MemFace/NR_inference_ref_images'
	final_out_folder = Path(final_out_folder)
	emoca, conf = load_model(path_to_models, model_name, mode)
	emoca.cuda()
	emoca.eval()
	ref_images = []
	
	vals = dict()
	vals["expcode"] = new_exp.to('cuda')
	# .unsqueeze(0).to('cuda')
	vals["shapecode"] = old_shape.to('cuda')
	vals["posecode"] = old_pose.to('cuda')
	vals["texcode"] =	old_tex.to('cuda')
	vals["cam"] = old_cam.to('cuda')
	vals["lightcode"] = torch.from_numpy(np.load('/home/avocoral/Downloads/Obamaset/Obama_vid_with_light/dataset_preprocessed/Obama_vid/000001_000/light.npy')).unsqueeze(0).repeat(300, 1, 1).to('cuda')
	vals["detailcode"] = old_detail.to('cuda')
	vals['detailemocode'] = None
	print(f'vals["expcode"].shape: {vals["expcode"].shape}')
	print(f'vals["posecode"].shape: {vals["posecode"].shape}')
	print(f'vals["shapecode"].shape: {vals["shapecode"].shape}')

	test_frames = ['/home/avocoral/Downloads/Obamaset/Obama_vid/Obama/006294_000/inputs.png']
	testdata = TestData(test_frames, iscrop=True, face_detector='fan')
	print(f"testdata[0]['image'].unsqueeze(0).shape: {testdata[0]['image'].unsqueeze(0).shape}")
	vals["images"] = testdata[0]['image'].unsqueeze(0).repeat(300, 1, 1, 1).to('cuda')

	vals, visdict = decode(emoca, vals, training=False)
	print(f"visdict['geometry_detail']: {len(visdict['geometry_detail'])}")	
	print(f"visdict['geometry_detail'].shape: {visdict['geometry_detail'].shape}")	
	for i, image in enumerate(visdict['geometry_detail'].view(300, 3, 224, 224)):
		imsave(final_out_folder / f"geometry_detail_{i}.png", _fix_image(torch_img_to_np(image)))
	frames_to_video('/home/avocoral/MemFace/NR_inference_ref_images', "/home/avocoral/MemFace/NR_ref_be.mp4", fps=30)
	
	return visdict['geometry_detail'].view(10, 30, 3, 224, 224)
	# imsave(final_out_folder / f"geometry_detail_{i}.png", _fix_image(torch_img_to_np(visdict['geometry_detail'][0])))
	# print(f'Frame {i} is ready!')
	# ref_images = torch.stack(ref_images)
	# return ref_images
# print(f'ref images are generated!')
# input_folder_path = '/home/avocoral/MemFace/test_folder'
# output_video_path = "Obama_NR_inference.mp4"
# output_audio_path = 'Obama_inference_15.wav'
# frames_to_video(input_folder_path, output_video_path, fps=30)
# print(f'ref video generated!')


if __name__ == '__main__':
# with dataset ref images - all works
# with ref images generated with create_ref_images() - nothing works
# visually look the same
# what other difference might they have?
	# audio_filepath = '/home/avocoral/MemFace/Obama_10.wav'
	# audio_embedding_filepath = '/home/avocoral/MemFace/Obama_10.pt'
	# extractAudioEncoding(audio_filepath, audio_embedding_filepath)
	# new_exp = audio2expression(audio_embedding_filepath)
	# new_exp = new_exp.to('cuda')
	# print(f'new_exp.shape: {new_exp.shape}')

	masked_imgs = get_masked_imgs()
	generated_ref_images = create_ref_images() * 255
	print(f'masked_imgs.shape: {masked_imgs.shape}')
	print(f'generated_ref_images.shape: {generated_ref_images.shape}')
	masked_ref_images = torch.cat([masked_imgs, generated_ref_images], dim=2).to('cuda:0')

	coeff_dir ='/home/avocoral/Downloads/Obamaset/Obama_vid/Obama'
	frame_num = len(os.listdir(coeff_dir))
	frame_names = [str(i).zfill(6)+'_000' for i in range(1, 301)]
	print(f'frame_names: {frame_names}')

	old_pose = []
	old_shape = []
	new_exp = []
	# frame_names
	for i, frame_name in enumerate(frame_names):
		old_pose.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'pose.npy'))))
		old_shape.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'shape.npy'))))
	old_pose = torch.stack(old_pose).to('cuda')
	old_shape = torch.stack(old_shape).to('cuda')

	for i, frame_name in enumerate([str(i).zfill(6)+'_000' for i in range(301, 601)]):
	 	new_exp.append(torch.from_numpy(np.load(os.path.join(coeff_dir, frame_name, 'exp.npy'))))

	new_exp = torch.stack(new_exp).to('cuda')

	landmarks3d = get_Om(old_pose, old_shape, new_exp, emoca=None, batch_size=0)
	landmarks3d = landmarks3d[:,48:,:].view(10, 30, 20, 3).to('cuda:0')
	print(f'landmarks3d loaded, landmarks3d.shape: {landmarks3d.shape}')


	device = 'cuda:0'
	NeuralRender = NeuralRender()
	checkpoints = torch.load("/home/avocoral/MemFace/checkpoints/MemFace/2vrf606v/checkpoints/epoch=13-step=37253.ckpt")
	NeuralRender.load_state_dict(checkpoints["state_dict"])
	NeuralRender.to(device)
	NeuralRender.eval()

	print('NR model loaded!')
	final_output = []
	with torch.no_grad():
		print(f'masked_ref_images.shape before inference: {masked_ref_images.shape}')
		print(f'landmarks3d.shape: {landmarks3d.shape}')
		output = NeuralRender(masked_ref_images, landmarks3d)
	print(f'output.shape: {output.shape}')


	# save final_output as images
	input_folder_path = "/home/avocoral/MemFace/NR_inference_final_output"
	output_video_path = "/home/avocoral/MemFace/NR_inference_be.mp4"
	os.makedirs(input_folder_path, exist_ok=True)
	for i, image_tensor in enumerate(output):
		image_np = (image_tensor.permute(1, 2, 0)).clamp(0, 255).byte().cpu().numpy()
		imageio.imwrite(os.path.join(input_folder_path, f"image_{i:04d}.png"), image_np)
	print('final images saved!')

	# create a video from final images
	frames_to_video(input_folder_path, output_video_path, fps=30)
	print(f'final video generated!')


