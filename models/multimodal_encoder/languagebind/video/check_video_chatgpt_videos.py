import torch
import cv2
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge('torch')
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

import os
import glob
from tqdm import tqdm

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def get_video_transform():
    # import pdb; pdb.set_trace()


    transform = Compose(
        [
            # UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ShortSideScale(size=224),
            CenterCropVideo(224),
            RandomHorizontalFlipVideo(p=0.5),
        ]
    )

    return transform

if __name__ == '__main__':
    directory = '/comp_robot/lushunlin/MotionGPT/video_datasets/videochatgpt/videochatgpt_tune'
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))
    # import pdb; pdb.set_trace()
    transform = get_video_transform()
    for video_path in tqdm(mp4_files):
        try:
            decord.bridge.set_bridge('torch')
            decord_vr = VideoReader(video_path, ctx=cpu(0))
            duration = len(decord_vr)
            frame_id_list = np.linspace(0, duration-1, 8, dtype=int)
            video_data = decord_vr.get_batch(frame_id_list)
            video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video_outputs = transform(video_data)
        except:
            with open('/comp_robot/lushunlin/MotionGPT/records/decord_error.txt', 'a') as f:
                f.write(video_path+'\n')