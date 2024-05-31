
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


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    config = config.vision_config
    # import pdb; pdb.set_trace()
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

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

    elif config.video_decode_backend == 'opencv':
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
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        # if '/comp_robot/lushunlin/MotionGPT/video_datasets/videochatgpt/videochatgpt_tune/v_BkBbzC6nIvA.mp4' == video_path \
        #     or '/comp_robot/lushunlin/MotionGPT/video_datasets/videochatgpt/videochatgpt_tune/v_fWVUEOVUzS4.mp4' == video_path:
        #     import  pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # try:
        
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        # duration = len(decord_vr)
        end_idx = len(decord_vr) - 1
        start_idx = 0

        
        if clip_end_sec is not None:
            fps = float(decord_vr.get_avg_fps())
            start_idx = max(start_idx, round(clip_start_sec * fps))
            end_idx = min(round(clip_end_sec * fps), end_idx)

        frame_id_list = np.linspace(start_idx, end_idx, num_frames, dtype=int)
        # import pdb; pdb.set_trace()
        video_data = decord_vr.get_batch(frame_id_list)

            
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)
        # import pdb; pdb.set_trace()
        # except:
        #     with open('/comp_robot/lushunlin/MotionGPT/records/decord_error.txt', 'a') as f:
        #         f.write(video_path+'\n')
        #     print(video_path)
        #     import pdb; pdb.set_trace()
            
            ###################

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f'video error at {video_path}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    # import pdb; pdb.set_trace()
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # self.config.vision_config.video_decode_backend = 'opencv'
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer
        

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, bound=None, **kwargs):
        if bound is not None:
            start = bound[0]
            end = bound[1]
        else:
            start = 0.0
            end = None
        # import pdb; pdb.set_trace()
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            # import pdb; pdb.set_trace()
            image_features = []
            for image in images:
                # try:
                image_features.append(self.image_processor(image, self.transform,video_decode_backend=self.config.vision_config.video_decode_backend, clip_start_sec=start, clip_end_sec=end, num_frames=self.config.vision_config.num_frames))
                # except:
                #     pass
            # image_features = [self.image_processor(image, self.transform,
            #                                        video_decode_backend=self.config.vision_config.video_decode_backend,
            #                                        num_frames=self.config.vision_config.num_frames) for image in images]
            # image_features = [torch.rand(3, 8, 224, 224) for image in images]
            # import pdb; pdb.set_trace()
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
