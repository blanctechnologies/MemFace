from loguru import logger
import numpy as np
import pickle

from PIL import Image, ImageDraw
import random
import torchvision

# from torchvision.io import read_image
# from torchvision.utils import save_image
# from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as F
import torch

# import openmesh as om
# import trimesh
# import sys
# from datasets import Audio2ExpDataModule

from emoca.gdl_apps.EMOCA.utils.load import load_model

# from emoca.gdl.utils.FaceDetector import FAN
from emoca.gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
from emoca.gdl_apps.EMOCA.utils.io import save_images, save_codes, test, decode
# from emoca.gdl_apps.EMOCA.utils.io import save_obj
import os
import shutil
from pathlib import Path
from tqdm import auto

# import cv2


from gdl.utils.lightning_logging import _fix_image
from skimage.io import imsave, imread

# from emoca.gdl.datasets.ImageTestDataset import TestData
from emoca.gdl.datasets.ImageDatasetHelpers import bbpoint_warp
# from emoca.gdl.datasets.ImageDatasetHelpers import point2bbox

# import math


def readReconstruction(filepath):
    f = np.load(filepath, allow_pickle=True)
    logger.info(f)
    logger.info(f.shape())
    for i in f:
        logger.info(f"i:{i}")
        logger.info(f"len of i:{len(i)}")


def readLandmarks(landmark_filepath, only_mouth=False, visualize=False):
    objects = []
    with open(landmark_filepath, "rb") as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    # logger.info(f'len = {len(objects[0])}')
    # mouth_landmarks = torch.Tensor([[[-int(objects[0][i][0]), -int(objects[0][i][1])] for i in range(len(objects[0])) if 48 <= i < 69]])
    if only_mouth:
        # mouth_landmarks = objects[1][48:69]
        mouth_landmarks = torch.Tensor(
            [[[int(objects[1][i][0]), int(objects[1][i][1])] for i in range(len(objects[1])) if 48 <= i < 69]]
        )
        inner_mouth_landmarks = [[objects[1][i][0], objects[1][i][1]] for i in range(len(objects[1])) if 60 <= i < 69]
        inner_mouth_landmarks.append(inner_mouth_landmarks[0])
        outer_mouth_landmarks = [[objects[1][i][0], objects[1][i][1]] for i in range(len(objects[1])) if 48 <= i < 60]
        outer_mouth_landmarks.append(outer_mouth_landmarks[0])
        logger.info(mouth_landmarks)
        logger.info(len(mouth_landmarks))
        return mouth_landmarks

    if visualize:
        # create black canvas
        vis_name = f'vis_{random.randint(100000,999999)}.jpg'
        # im = Image.new('RGB', (256, 256), (0, 0, 0))
        original_image = Image.open(
            r'/home/avocoral/MemFace/emoca/output/processed_2023_Jan_02_17-22-45/testvid/videos/000042.png'
        )

        original_image.save('orig_test.jpg', quality=95)
        img = torchvision.transforms.functional.to_tensor(original_image).cuda()
        # im.save(vis_name, quality=95)
        # draw mouth landmarks
        # img = pil_to_tensor(im)
        # img = read_image(vis_name)
        # keypoints = torch.Tensor([mouth_landmarks])
        # logger.info(keypoints)

        # Mouth connections 
        # outer = [(i, i + 1) for i in range(9)] + [(9, 0)]
        # mouth_connections = inner + outer
        # logger.info(mouth_connections)
        
        # draw_keypoints(img, keypoints, colors='white', connectivity = mouth_connections, radius=1, width=1)
        # .to(torch.uint8)
        logger.info(img.to(torch.uint8))
        final_image_test = F.to_pil_image(img.to(torch.uint8))
        final_image_test.save('final_image_test.jpg', quality=95)
        # res = draw_keypoints(img.to(torch.uint8), keypoints, colors='white', connectivity = mouth_connections, radius=0, width=1)
        # final_image = F.to_pil_image(res)
        # vis_name = f'vis_{random.randint(100000,999999)}.jpg'
        # final_image.save(vis_name, quality=95)
        # save_image(final_image, vis_name)
        logger.info(f'visualization filename: {vis_name}')
    return objects
    # return mouth_landmarks


def readCoeff(shape_filepath, exp_filepath, pose_filepath, cam_filepath):
    logger.info('+----------- Exp ------------+')
    exp = np.load(exp_filepath, allow_pickle=True)
    logger.info(torch.from_numpy(exp))
    logger.info(f"len: {len(exp)}")

    logger.info('+----------- Pose ------------+')
    pose = np.load(pose_filepath, allow_pickle=True)
    logger.info(torch.from_numpy(pose))
    logger.info(f"len: {len(pose)}")

    logger.info('+----------- Shape ------------+')
    shape = np.load(shape_filepath, allow_pickle=True)
    logger.info(shape)
    logger.info(f"len: {len(shape)}")

    logger.info('+----------- Cam ------------+')
    cam = np.load(cam_filepath, allow_pickle=True)
    logger.info(cam)
    logger.info(f"len: {len(cam)}")

    return shape, exp, pose, cam


def readObj(obj_filepath):
    # mesh = om.read_trimesh(obj_filepath)
    # with open(obj_filepath) as f:
    #   lines = f.readlines()
    # vertices = [line for line in lines if line.startswith('v ')]
    # faces = [line for line in lines if line.startswith('f ')]
    mesh = trimesh.load_mesh(obj_filepath)
    logger.info('+----------- Mesh ------------+')
    logger.info(mesh.vertices)
    logger.info(f'len: {mesh.vertices.shape}')
    return None


def move_files_around(coeff_dir='/mnt/sda/AVSpeech/dataset_v2_video', metadata_dir='/mnt/sda/AVSpeech/dataset_v2_meta'):
    faulty_vid_list = []
    faulty_vid_counter = 0
    for filename in os.listdir(coeff_dir):
        try:
            # move metadata.pkl
            shutil.move(
                os.path.join(coeff_dir, filename, 'metadata.pkl'),
                os.path.join(coeff_dir, filename, filename, 'metadata.pkl'),
            )
            # move folder with all metadata outside
            shutil.move(os.path.join(coeff_dir, filename, filename), os.path.join(metadata_dir, filename))
        except Exception as e:
            faulty_vid_counter += 1
            faulty_vid_list.append(filename)
    logger.info(f'number of faulty vids, that werent moved: {faulty_vid_counter}')
    logger.info(f'faulty_vid_list: {faulty_vid_list}')


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
    dataset_dir = '/mnt/sda/AVSpeech/NR_v1_dataset/dataset_v1'
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
    path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
    model_name = 'EMOCA_v2_lr_mse_20'
    mode = 'detail'

    if not emoca:
        emoca, conf = load_model(path_to_models, model_name, mode)
        # emoca.cuda()
        emoca.eval()

    if batch_size != 0:
        pose = torch.reshape(pose, (pose.shape[0] * pose.shape[1], pose.shape[2]))
        shape = torch.reshape(shape, (shape.shape[0] * shape.shape[1], shape.shape[2]))
        exp = torch.reshape(exp, (exp.shape[0] * exp.shape[1], exp.shape[2]))
        logger.info(f'pose.shape: {pose.shape}')
        logger.info(f'shape.shape: {shape.shape}')
        logger.info(f'exp.shape: {exp.shape}')
        verts, landmarks2d, landmarks3d_hat, _ = emoca.deca.flame(
            shape_params=shape, expression_params=exp, pose_params=pose
        )
        logger.info(f'landmarks3d_hat.shape after get_Om: {landmarks3d_hat.shape}')
        sequence_length = landmarks3d_hat.shape[0] // batch_size
        landmarks3d_hat = landmarks3d_hat.view(
            batch_size, sequence_length, landmarks3d_hat.shape[1], landmarks3d_hat.shape[2]
        )
        logger.info(f'landmarks3d_hat.shape batched: {landmarks3d_hat.shape}')
        landmarks3d_hat = landmarks3d_hat[:, :, 48:, :]

    else:
        verts, landmarks2d, landmarks3d_hat, _ = emoca.deca.flame(
            shape_params=shape, expression_params=exp, pose_params=pose
        )
        logger.info(f'landmarks3d_hat.shape: {landmarks3d_hat.shape}')
        landmarks3d_hat = landmarks3d_hat[:, 48:, :]

    return landmarks3d_hat


def neural_rendering_facereconstruction(filepath):
    path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
    input_video = filepath
    output_folder = os.path.join(filepath.rsplit('.', 1)[0], 'dataset_preprocessed')
    model_name = 'EMOCA_v2_lr_mse_20'
    image_type = 'geometry_detail'
    cat_dim = 0
    include_transparent = False
    processed_subfolder = None
    mode = 'detail'

    dm = TestFaceVideoDM(
        input_video, output_folder, processed_subfolder=Path(input_video).stem, batch_size=1, num_workers=24
    )
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
    for j, batch in enumerate(auto.tqdm(dl)):
        current_bs = batch["image"].shape[0]
        img = batch
        vals, visdict = test(emoca, img)
        # logger.info('+---------------- Landmarks3D ----------------+')
        # logger.info(vals['landmarks3d'])
        for i in range(current_bs):
            # name = f"{(j*batch_size + i):05d}"
            name = batch["image_name"][i]

            sample_output_folder = Path(outfolder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)
            # break
            save_images(outfolder, name, visdict, i)
            save_codes(Path(outfolder), name, vals, i)


def find_most_similar_tensors(K_nr):
    K_nr_landmarks = K_nr[:, :20]
    num_tensors = K_nr_landmarks.size(0)
    K_nr_reshaped = K_nr_landmarks.view(num_tensors, -1)
    pairwise_distances = torch.cdist(K_nr_reshaped, K_nr_reshaped, p=2).pow(2)
    # torch.fill_diagonal_(pairwise_distances, float('inf'))
    pairwise_distances[range(pairwise_distances.size(0)), range(pairwise_distances.size(1))] = float('inf')
    min_indices = torch.argmin(pairwise_distances)
    row_idx = min_indices // num_tensors
    col_idx = min_indices % num_tensors
    rms_distance = torch.sqrt(pairwise_distances[row_idx, col_idx])
    # return K_nr[row_idx], K_nr[col_idx], rms_distance
    return row_idx, col_idx, rms_distance


def copy_Knr_and_replace(K_nr, k_idx, k_tmp):
    K_nr_clone = K_nr.clone()
    K_nr_clone[k_idx] = k_tmp
    return


def construct_explicitmem():
    N = 300
    K_nr_filepath = '/home/avocoral/Downloads/Obamaset/K_nr.pt'
    V_nr_filepath = '/home/avocoral/Downloads/Obamaset/V_nr.pt'

    if os.path.exists(K_nr_filepath) and os.path.exists(V_nr_filepath):
        K_nr = torch.load(K_nr_filepath)
        V_nr = torch.load(V_nr_filepath)
        return K_nr, V_nr

    # Step 0: Build K_all, and V_all will be just the last 6 numbers in each tensor of K_all
    frame_names = os.listdir(data_dir)
    frame_names.sort()
    frame_num = len(frame_names)
    K_all = torch.zeros((frame_num, 66))
    logger.info(f'first 30 images: {frame_names[:30]}')
    for i, frame in enumerate(frame_names):
        label = torch.tensor([int(digit) for digit in frame[:6]])
        landmarks3d_path = os.path.join(data_dir, frame, 'landmarks3d.npy')
        landmarks3d = torch.from_numpy(np.load(landmarks3d_path))[:, 48:, :].squeeze(0).reshape(60)
        logger.info(f'landmarks3d.shape: {landmarks3d.shape}')
        logger.info(f'landmarks3d: {landmarks3d}')
        K_all[i] = torch.cat((landmarks3d, label), dim=0)

    # step 1: Initialize K_nr, V_nr
    permutation_indices = torch.randperm(frame_num)
    K_nr = K_all[permutation_indices[:N]]
    # step 2: Find two most similar mouth shapes:
    k_m1_idx, k_m2_idx, Dmin = find_most_similar_tensors(K_nr)
    logger.info(f'k_m1_idx: {k_m1_idx}')
    logger.info(f'k_m2_idx: {k_m2_idx}')
    logger.info(f'Dmin: {Dmin}')
    # step 3: go through all tensors in K_all
    i = 0
    for i, k_tmp in enumerate(K_all):
        logger.info(f"Preprocessed {i}/{len(K_all)}")
        # create a copy of K_nr where k_m1 replaced with k_tmp
        K_tmp1 = K_nr.clone()
        K_tmp1[k_m1_idx] = k_tmp
        # find two most similar tensors in K_nr and their distance Dmin1
        # logger.info(f'K_tmp1[0]: {K_tmp1[0]}')
        _, _, Dmin1 = find_most_similar_tensors(K_tmp1)
        # create a copy of K_nr where k_m2 replaced with k_tmp
        K_tmp2 = K_nr.clone()
        K_tmp2[k_m2_idx] = k_tmp
        # find two most similar tensors in K_nr and their distance Dmin1
        _, _, Dmin2 = find_most_similar_tensors(K_tmp2)

        if max(Dmin1, Dmin2) > Dmin:
            if Dmin1 > Dmin2:
                K_nr[k_m1_idx] = k_tmp
            else:
                K_nr[k_m2_idx] = k_tmp
        k_m1_idx, k_m2_idx, Dmin = find_most_similar_tensors(K_nr)

    V_nr_new_names = [f'{"".join(map(str, map(int, elem[-6:].tolist())))}.png' for elem in K_nr]
    image_path = '/home/avocoral/Downloads/Obamaset/Obama_meta/cropped_frames/{}'
    V_nr_new = torch.stack([load_tensor_image(image_path.format(image_name)) for image_name in V_nr_new_names])
    K_nr_new = K_nr[:, :60]
    logger.info(f'K_nr_new.shape: {K_nr_new.shape}')
    logger.info(f'V_nr_new.shape: {V_nr_new.shape}')

    # save K_nr_new and V_nr_new
    torch.save(K_nr_new, K_nr_filepath)
    torch.save(V_nr_new, V_nr_filepath)

    return K_nr_new, V_nr_new


def load_tensor_image(image_name):
    with Image.open(image_name) as image:
        # only load the lower part of the image with lips
        logger.info(f'image.size[1]: {image.size[1]}')
        logger.info(f'image.size[1]//2: {image.size[1]//2}')
        image = image.crop((0, image.size[1] // 2, image.size[0] - 1, image.size[1] - 1))
        image = image.convert("RGB")
        tensor_image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor_image = tensor_image.view(image.size[1], image.size[0], -1)
        tensor_image = tensor_image.permute(2, 0, 1).float().div(255.0)
    return tensor_image


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)


def create_reconstruction_from_vals(
    coeff_folderpath, final_output_folder, new_exp=None, emoca=None, start_frame=None, end_frame=None, resolution=451
):
    path_to_models = "/home/avocoral/MemFace/emoca/assets/EMOCA/models"
    model_name = 'EMOCA_v2_lr_mse_20'
    mode = 'detail'

    # coeff_folderpath = '/home/avocoral/Downloads/Obamaset/Obama_vid/Obama'

    # final_out_folder = '/home/avocoral/MemFace/test_folder'
    # final_out_folder = Path(final_out_folder)

    # write custom resolution to the config file
    if resolution != None:
        with open('/home/avocoral/MemFace/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg.yaml', 'r') as file:
            lines = file.readlines()

        new_line = f"    image_size: {resolution}"
        lines[285] = new_line + '\n'
        with open('/home/avocoral/MemFace/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg.yaml', 'w') as file:
            file.writelines(lines)

    if emoca == None:
        emoca, conf = load_model(path_to_models, model_name, mode)
        emoca.cuda()
        emoca.eval()

    frame_names = os.listdir(coeff_folderpath)
    frame_names.sort()
    num_frames = len(frame_names)
    for i, frame_name in enumerate(frame_names):
        if i < start_frame or i > end_frame:
            continue
        logger.info(f'---------- #{i-start_frame}/{start_frame-end_frame} -----------')
        vals = dict()
        if new_exp != None:
            vals["expcode"] = new_exp[i - start_frame - 1].unsqueeze(0).to('cuda')
            logger.info(f'vals["expcode"].shape: {vals["expcode"].shape}')
        else:
            vals["expcode"] = (
                torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'exp.npy'))).unsqueeze(0).to('cuda')
            )

        vals["posecode"] = (
            torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'pose.npy'))).unsqueeze(0).to('cuda')
        )
        vals["shapecode"] = (
            torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'shape.npy'))).unsqueeze(0).to('cuda')
        )
        vals["texcode"] = (
            torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'tex.npy'))).unsqueeze(0).to('cuda')
        )
        vals["cam"] = (
            torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'cam.npy'))).unsqueeze(0).to('cuda')
        )
        vals["detailcode"] = (
            torch.from_numpy(np.load(os.path.join(coeff_folderpath, frame_name, 'detail.npy'))).unsqueeze(0).to('cuda')
        )
        vals["lightcode"] = (
            torch.from_numpy(
                np.load(
                    '/home/avocoral/Downloads/Obamaset/Obama_vid_with_light/dataset_preprocessed/Obama_vid/000001_000/light.npy'
                )
            )
            .unsqueeze(0)
            .to('cuda')
        )
        vals['detailemocode'] = None
        vals["images"] = torch.randn((3, resolution, resolution)).unsqueeze(0).to('cuda')

        logger.info(f'vals["expcode"].shape: {vals["expcode"].shape}')
        logger.info(f'vals["posecode"].shape: {vals["posecode"].shape}')
        logger.info(f'vals["shapecode"].shape: {vals["shapecode"].shape}')

        vals, visdict = decode(emoca, vals, training=False)
        logger.info(f"visdict['geometry_detail'][0].shape: {visdict['geometry_detail'][0].shape}")

        imsave(
            Path(final_output_folder) / f"geometry_detail_{frame_name}.png",
            _fix_image(torch_img_to_np(visdict['geometry_detail'][0])),
        )


def crop_out_inference_frames(videos_folderpath=None, cropped_frames_folderpath=None, bboxes_filepath=None):
    videos_folderpath = '/home/avocoral/Downloads/Obamaset/Obama_meta/videos'
    landmarks_folderpath = '/home/avocoral/Downloads/Obamaset/Obama_meta/landmarks'
    new_landmarks_folderpath = '/home/avocoral/Downloads/Obamaset/Obama_meta/new_landmarks'
    cropped_frames_folderpath = '/home/avocoral/Downloads/Obamaset/Obama_meta/cropped_frames'
    bboxes_filepath = '/home/avocoral/Downloads/Obamaset/Obama_meta/detections/bboxes.pkl'
    # landmarks = [detection_fnames, landmark_fnames, centers, sizes, last_frame_id]
    bboxes = readLandmarks(bboxes_filepath)
    min_index = max(range(len(bboxes[2])), key=lambda i: bboxes[2][i][0])
    min_size = bboxes[2][min_index][0]
    num_frames = len(os.listdir(videos_folderpath))
    logger.info(f'min_size = {min_size}')
    videos_folderpath_list = os.listdir(videos_folderpath).sort()
    cropped_warped_landmarks_list = []
    for i, fname in enumerate(bboxes[4]):
        # double check center, name and fname
        logger.info(f'----------- frame# {i}/{num_frames} -----------')
        detection_fname = str(bboxes[0][i][0]).split('/', -1)[-1].split('_')[0] + '.png'
        logger.info(f'detection_fname: {detection_fname}')
        img = imread(os.path.join(videos_folderpath, detection_fname))
        # img = Image.open(os.path.join(videos_folderpath, detection_fname))
        # width, height = img.shape[0], img.shape[1]
        center = bboxes[1][i][0]
        size = bboxes[2][i][0]
        landmarks_filepath = os.path.join(landmarks_folderpath, str(bboxes[4][i][0]).split('/')[-1])
        # logger.info(f'landmarks_filepath:{landmarks_filepath}, orig_fname: {bboxes[4][i][0]}')
        landmarks = readLandmarks(landmarks_filepath)[1]
        # logger.info(f'landmarks: {landmarks}')
        # logger.info(f'landmarks.type: {landmarks.type}')
        logger.info(f'center: {center}')
        # point2bbox(center, size)
        # left = center[0] - max_size // 2
        # top = center[1] - max_size // 2
        # right = center[0] + max_size // 2
        # bottom = center[1] + max_size // 2
        ### cropping and warping ###
        # get mask (hacky way)
        # dst_image, dts_landmark = bbpoint_warp(image, center, size, self.image_size, landmarks=landmarks[bi])
        # cropped_warped_img = bbpoint_warp(img, center, size, min_size, output_shape=(img.shape[0], img.shape[1]), inv=False)
        cropped_warped_img, cropped_warped_landmarks = bbpoint_warp(img, center, size, min_size, landmarks=landmarks)
        # cropped_warped_img = Image.fromarray((warped_im * 255).astype(np.uint8))

        imsave(os.path.join(cropped_frames_folderpath, detection_fname), cropped_warped_img)
        file_path = os.path.join(new_landmarks_folderpath, f'{detection_fname[:-4]}_000.pkl')
        logger.info(f'file_path: {file_path}')
        with open(file_path, 'wb') as file:
            pickle.dump(cropped_warped_landmarks, file)
    # save cropped_warped_landmarks_list
    # move cropped_warped_imgs to gpu
    # np.save("/home/avocoral/Downloads/Obamaset/Obama_meta/processed_landmarks", cropped_warped_landmarks_list)
    # we don't need new 2D landmarks, we use world landmarks and they are the same independent of image cropping

    #######
    # cropped_img = img.crop((left, top, right, bottom))
    # cropped_warped_img.save(os.path.join(cropped_frames_folderpath, fname))


if __name__ == '__main__':
    # landmarks = readLandmarks('/home/avocoral/Downloads/Obamaset/Obama_meta/new_landmarks/002625_000.pkl')
    # logger.info(f'new_landmarks:{landmarks}')
    # crop_out_inference_frames()
    # create_reconstruction_from_vals()
    # landmarks_filepath = '/home/avocoral/Downloads/Obamaset/Obama_meta/detections/bboxes.pkl'
    construct_explicitmem('/home/avocoral/Downloads/Obamaset/Obama_vid/Obama')
    # landmarks = [detection_fnames, landmark_fnames, centers, sizes, last_frame_id]
    # landmarks = readLandmarks(landmarks_filepath)
    # exp_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/exp.npy'
    # pose_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/pose.npy'
    # shape_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/shape.npy'
    # cam_filepath = '/mnt/sda/AVSpeech/video/GWwK4ak096M_9/000001_000/cam.npy'
    # shapecode, expcode, posecode, camcode = readCoeff(shape_filepath, exp_filepath, pose_filepath, cam_filepath)
    # logger.info(f'dim of shape: {torch.from_numpy(shapecode).ndimension()}')
    # logger.info(f'dim of exp: {torch.from_numpy(expcode).ndimension()}')
    # betas = torch.cat([torch.from_numpy(shapecode).unsqueeze(0), torch.from_numpy(expcode).unsqueeze(0)], dim=1)
    # logger.info(f'betas: {betas}')
    # landmarks3d = get_Om(posecode, shapecode, expcode)
    # logger.info(landmarks3d)
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
    # logger.info(f'landmarks_hat.shape: {landmarks3d_hat.shape}')

    # --- preprocess NR dataset ---
    # dataset_dir = '/mnt/sda/AVSpeech/NR_v1_dataset/dataset_v1_videos'
    #
    # for i, video_name in enumerate(os.listdir(dataset_dir)):
    #       logger.info(f'! video #{i} preprocessed !')
    #       filepath = os.path.join(dataset_dir, video_name)
    #
    # filepath = '/home/avocoral/Downloads/Obamaset/Obama_vid.mp4'
    # neural_rendering_facereconstruction(filepath)

    # K_nr, V_nr = construct_explicitmem()
    # logger.info(f'K_nr.shape: {K_nr.shape}')
    # logger.info(f'V_nr.shape: {V_nr.shape}')
    # logger.info(find_optimal_lipsbox())
    # move_files_around_nr()
    # move_files_some_more()
