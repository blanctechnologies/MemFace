# 1. adaptation - ommit for now
# 2. inference
# 3. pasting generated images back
# 4. merge frames
# 5. add audio
# 6. cleanup
import tempfile
import os
from neuralrendering import *
from neuralrendering import NeuralRender
from datasets import NeuralRenderingDataset
import os
import torch
from PIL import Image
from moviepy.editor import VideoFileClip
import imageio
import shutil

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

from emoca.gdl.datasets.ImageDatasetHelpers import point2bbox, bbpoint_warp
from neuralrendering import NeuralRender
from neuralrendering import *
from utils import readLandmarks

def adaptation():
	pass


def get_masked_imgs():
	masked_images = []
	NRdataset = NeuralRenderingDataset()
	for i in range(12):
		masked_ref_image, _, _  = NRdataset.__getitem__(i*25)
		masked_images.append(masked_ref_image[:, :3, :, :])
		
	masked_images = torch.stack(masked_images).view(12, 25, 3, 450, 450).to('cuda')
	# save dataset ref images
	return masked_images


def create_ref_images_and_landmarks():
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
	for i, frame_name in enumerate([str(i).zfill(6)+'_000' for i in range(900, 1200)]):
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
	output_images = []
	landmarks3d = []
	for i in range(new_exp.shape[0] // 25):
		vals = dict()
		vals["expcode"] = new_exp[i*25:(i+1)*25, :].to('cuda')
		# .unsqueeze(0).to('cuda')
		vals["shapecode"] = old_shape[i*25:(i+1)*25, :].to('cuda')
		vals["posecode"] = old_pose[i*25:(i+1)*25, :].to('cuda')
		vals["texcode"] =	old_tex[i*25:(i+1)*25, :].to('cuda')
		vals["cam"] = old_cam[i*25:(i+1)*25, :].to('cuda')
		vals["lightcode"] = torch.from_numpy(np.load('/home/avocoral/Downloads/Obamaset/Obama_vid_with_light/dataset_preprocessed/Obama_vid/000001_000/light.npy')).unsqueeze(0).repeat(25, 1, 1).to('cuda')
		vals["detailcode"] = old_detail[i*25:(i+1)*25, :].to('cuda')
		vals['detailemocode'] = None
		print(f'vals["expcode"].shape: {vals["expcode"].shape}')
		print(f'vals["posecode"].shape: {vals["posecode"].shape}')
		print(f'vals["shapecode"].shape: {vals["shapecode"].shape}')

		# test_frames = ['/home/avocoral/Downloads/Obamaset/Obama_vid/Obama/006294_000/inputs.png']
		# testdata = TestData(test_frames, iscrop=True, face_detector='fan')
		# print(f"testdata[0]['image'].unsqueeze(0).shape: {testdata[0]['image'].unsqueeze(0).shape}")
		vals["images"] = torch.randn(25, 3, 450, 450).to('cuda')
		vals, visdict = decode(emoca, vals, training=False)
		print(f"visdict['geometry_detail']: {len(visdict['geometry_detail'])}")	
		print(f"visdict['geometry_detail'].shape: {visdict['geometry_detail'].shape}")
		output_images.append(visdict['geometry_detail'])
		landmarks3d_tmp = get_Om(vals["posecode"], vals["shapecode"], vals["expcode"], emoca=emoca, batch_size=0)
		landmarks3d.append(landmarks3d_tmp)
	output_images = torch.stack(output_images)
	landmarks3d = torch.stack(landmarks3d)
	print(f'output_images.shape: {output_images.shape}')
	print(f'landmarks3d.shape: {landmarks3d.shape}')
	# for i, image in enumerate(visdict['geometry_detail'].view(300, 3, 224, 224)):
		# imsave(final_out_folder / f"geometry_detail_{i}.png", _fix_image(torch_img_to_np(image)))
	# frames_to_video('/home/avocoral/MemFace/NR_inference_ref_images', "/home/avocoral/MemFace/NR_ref_be.mp4", fps=30)
	
	torch.cuda.empty_cache()	
	
	return output_images, landmarks3d

def inference(masked_images, ref_images, landmarks3d, final_video_path, final_audio_path):
	import neuralrendering
	device = 'cuda:0'
	NeuralRender = neuralrendering.NeuralRender()
	checkpoints = torch.load("/home/avocoral/MemFace/checkpoints/nr_450.ckpt")
	NeuralRender.load_state_dict(checkpoints["state_dict"])
	NeuralRender.to(device)
	NeuralRender.eval()

	print('NR model loaded!')
	final_output = []
	masked_images.to('cuda')
	ref_images.to('cuda')
	ref_images = ref_images * 255
	landmarks3d.to('cuda')
	masked_ref_images = torch.cat([masked_images, ref_images], dim=2).to('cuda:0')

	# merge masked_images and ref_images
	for i in range(masked_ref_images.shape[0]):
		with torch.no_grad():
			print(f'masked_ref_images[i].unsqueeze(0).shape before inference: {masked_ref_images[i].unsqueeze(0).shape}')
			print(f'landmarks3d[i].unsqueeze(0).shape: {landmarks3d[i].unsqueeze(0).shape}')
			output = NeuralRender(masked_ref_images[i].unsqueeze(0), landmarks3d[i].unsqueeze(0))
			final_output.append(output)
	final_output = torch.stack(final_output, dim=0).view(300, 3, 450, 450)
	print(f'final_output.shape: {final_output.shape}')
	output.to('cuda')
	torch.cuda.empty_cache()

	# save final_output as images
	gen_images_folderpath = tempfile.mkdtemp(dir='/home/avocoral/MemFace')
	for i, image_tensor in enumerate(final_output):
		image_np = (image_tensor.permute(1, 2, 0)).clamp(0, 255).byte().cpu().numpy()
		imageio.imwrite(os.path.join(gen_images_folderpath, f"image_{i:04d}.png"), image_np)
	print('final images saved!')

	# pasting generated images into original frames	
	videos_folderpath = '/home/avocoral/Downloads/Obamaset/Obama_meta/videos'
	bboxes_filepath = '/home/avocoral/Downloads/Obamaset/Obama_meta/detections/bboxes.pkl'
	# landmarks = [detection_fnames, landmark_fnames, centers, sizes, last_frame_id]
	bboxes = readLandmarks(bboxes_filepath)
	gen_image_fnames = os.listdir(gen_images_folderpath)
	gen_image_fnames.sort()
	final_images_folderpath = tempfile.mkdtemp(dir='/home/avocoral/MemFace')
	masked_images = masked_images.view(masked_images.shape[0]*masked_images.shape[1], 3, 450, 450)
	
	im_list = []

	# paste generated pictures into original frames
	for i, fname in enumerate(bboxes[4]):
		if i >= masked_images.shape[0]:
			break
		center = bboxes[1][i][0]
		size = bboxes[2][i][0]
		print(f'bboxes[0][i][0]: {bboxes[0][i][0]}')
		original_fname = str(bboxes[0][i][0]).split('/', -1)[-1].split('_')[0] + '.png'
		orig_img = np.array(Image.open(os.path.join(videos_folderpath, original_fname)))
		
		# creating mask
		print(f'i:{i}')
		print(f'masked_images.shape: {masked_images.shape}')
		mask_im = np.transpose(masked_images[i].cpu().numpy(), (1, 2, 0))
		vis_mask = 255 - (np.prod(mask_im, axis=2) > 255).astype(np.uint8) * 255

		print(f'orig_img filepath: {os.path.join(videos_folderpath, original_fname)}')
		print(f'generated_img filepath: {os.path.join(gen_images_folderpath, gen_image_fnames[i])}')
		generated_img = np.array(Image.open(os.path.join(gen_images_folderpath, gen_image_fnames[i])))
		print(f'generated_img.shape : {generated_img.shape}')
		print(f'center, size = {center}, {size}')
		print(f'orig_img.shape[0] = {orig_img.shape[0]}')
		print(f'orig_img.shape[1] = {orig_img.shape[1]}')
		warped_im = bbpoint_warp(generated_img, [center[0]*1, center[1]*1], size*1, generated_img.shape[0], output_shape=(orig_img.shape[0], orig_img.shape[1]), inv=False)
		print(f'vis_mask.shape: {vis_mask.shape}')
		warped_mask = bbpoint_warp(vis_mask, [center[0]*1, center[1]*1], size*1, generated_img.shape[0], output_shape=(orig_img.shape[0], orig_img.shape[1]), inv=False)
		# warped_im = bbpoint_warp(generated_img, center, 1.25 * size, orig_img.shape[0], output_shape=(orig_img.shape[0], orig_img.shape[1]), inv=False)
		# print(f'vis_mask.shape: {vis_mask.shape}')
		# warped_mask = bbpoint_warp(vis_mask, center, 1.25 * size, orig_img.shape[0], output_shape=(orig_img.shape[0], orig_img.shape[1]), inv=False)
		print(f'warped_mask.shape: {warped_mask.shape}')
		vis_pil = Image.fromarray((warped_im * 255).astype(np.uint8))
		mask_pil = Image.fromarray((warped_mask * 255).astype(np.uint8))

		orig_img = Image.fromarray(orig_img)
		orig_copy = orig_img.copy()
		orig_copy.paste(vis_pil, (0, 0), mask_pil)
		orig_copy.save(os.path.join(final_images_folderpath, f"image_{i:04d}.png"))
		orig_copy.convert("RGB")
		im_list.append(orig_copy)
	
	# merge frames into video
	temp_video_file = tempfile.mktemp(suffix=".mp4", dir='/home/avocoral/MemFace')
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	writer = cv2.VideoWriter(temp_video_file, fourcc, 29.97, (im_list[0].width, im_list[0].height))	
	for im in im_list:
		frame = np.array(im)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		writer.write(frame)
		# merge frames into video
		# cat_dim = 1
		# im = np.concatenate(im_list, axis=cat_dim)
		# im_cv = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
		# writer.write(im_cv)
	writer.release()
	
	# add audio to the video
	ffmpeg_command = f"ffmpeg -y -i {temp_video_file} -i {final_audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {final_video_path}"
	subprocess.Popen(ffmpeg_command, shell=True)
	
	# clean up
	print(f'temp_video_file: {temp_video_file}')
	if os.path.exists(temp_video_file):
		os.remove(temp_video_file)
		# print(f'exists!')
	if os.path.exists(final_images_folderpath):
		shutil.rmtree(final_images_folderpath)
		




if __name__ == "__main__":
	final_video_path = '/home/avocoral/MemFace/NR_inference_be.mp4'
	final_audio_path = '/home/avocoral/Downloads/Obamaset/Obama_vid_30_40.wav'
	ref_images, landmarks3d = create_ref_images_and_landmarks()
	masked_images = get_masked_imgs()
	inference(masked_images, ref_images, landmarks3d, final_video_path, final_audio_path)
	
	
