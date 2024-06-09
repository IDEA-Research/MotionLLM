import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional
from typing import Dict, List, Literal, Optional, Tuple
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable

import lightning as L
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import models.vqvae as vqvae
from generate import generate
from lit_llama import Tokenizer, LLaMA, LLaMAConfig
from lit_llama.lora import lora
from lit_llama.utils import EmptyInitOnDevice
from lit_gpt.utils import lazy_load
from scripts.video_dataset.prepare_video_dataset_video_llava import generate_prompt_mlp
from options import option
import imageio
from tqdm import tqdm
from models.multimodal_encoder.builder import build_image_tower, build_video_tower
from models.multimodal_projector.builder import build_vision_projector

warnings.filterwarnings('ignore')

args = option.get_args_parser()


class LlavaMetaModel:

    def __init__(self, config, pretrained_checkpoint):
        super(LlavaMetaModel, self).__init__()
        # import pdb; pdb.set_trace()
        if hasattr(config, "mm_image_tower") or hasattr(config, "image_tower"):
            self.image_tower = build_image_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_video_tower") or hasattr(config, "video_tower"):
            self.video_tower = build_video_tower(config, delay_load=True)
            # import pdb; pdb.set_trace()
            self.mm_projector = build_vision_projector(config)
            self.load_video_tower_pretrained(pretrained_checkpoint)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower


    def get_all_tower(self, keys):
        tower = {key: getattr(self, f'get_{key}_tower') for key in keys}
        return tower


    def load_video_tower_pretrained(self, pretrained_checkpoint):
        self.mm_projector.load_state_dict(pretrained_checkpoint, strict=True)


    def initialize_image_modules(self, model_args, fsdp=None):
        image_tower = model_args.image_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_image_tower = image_tower

        image_tower = build_image_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.image_tower = [image_tower]
        else:
            self.image_tower = image_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = image_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_video_modules(self, model_args, fsdp=None):
        video_tower = model_args.video_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_video_tower = video_tower

        video_tower = build_video_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.video_tower = [video_tower]
        else:
            self.video_tower = video_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = video_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def encode_images(self, images):
        image_features = self.get_image_tower()(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def encode_videos(self, videos):
        video_features = self.get_video_tower()(videos) # torch.Size([1, 2048, 1024])
        video_features = self.mm_projector(video_features.float()) # torch.Size([1, 2048, 4096])

        return video_features
    
    def get_multimodal_embeddings(self, X_modalities):
        Xs, keys= X_modalities
        X_features = getattr(self, f'encode_{keys[0]}s')(Xs)  # expand to get batchsize

        return X_features
    
def get_processor(X, config, device, pretrained_checkpoint_tower, model_path = 'LanguageBind/Video-LLaVA-7B'):


    processor = {}

    mm_backbone_mlp_model = LlavaMetaModel(config, pretrained_checkpoint_tower)

    print(X)    
    if 'Image' in X:
        image_tower = mm_backbone_mlp_model.get_image_tower() # LanguageBindImageTower()
        if not image_tower.is_loaded:
            image_tower.load_model()
        image_tower.to(device=device, dtype=torch.float16)
        image_processor = image_tower.image_processor
        processor['image'] = image_processor

    if 'Video' in X:
        video_tower = mm_backbone_mlp_model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_tower.to(device=device, dtype=torch.float16)
        video_processor = video_tower.video_processor
        processor['video'] = video_processor

    return mm_backbone_mlp_model, processor


class Projection(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear_proj = nn.Linear(512, 4096)
    def forward(self, x):
        return self.linear_proj(x)


class ProjectionNN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(512, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096)
        )
    def forward(self, x):
        return self.proj(x)


def main(
    quantize: Optional[str] = None,
    dtype: str = "float32",
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.8,
    accelerator: str = "auto",
) -> None:
    
    # import pdb; pdb.set_trace()
    lora_path = Path(args.lora_path)
    pretrained_llm_path = Path(f"./checkpoints/vicuna-7b-v1.5/lit_model.pth")
    tokenizer_llm_path = Path("./checkpoints/vicuna-7b-v1.5/tokenizer.model")
    
    # assert lora_path.is_file()
    assert pretrained_llm_path.is_file()
    assert tokenizer_llm_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    t0 = time.time()
    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ), lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, enabled=True):
        checkpoint_dir = Path("checkpoints/vicuna-7b-v1.5")
        lora_query = True
        lora_key = False
        lora_value = True
        lora_projection = False
        lora_mlp = False
        lora_head = False
        config = Config.from_name(
            name=checkpoint_dir.name,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            to_query=lora_query,
            to_key=lora_key,
            to_value=lora_value,
            to_projection=lora_projection,
            to_mlp=lora_mlp,
            to_head=lora_head,
        )
        model = GPT(config).bfloat16()
    
    mlp_path = args.mlp_path
    pretrained_checkpoint_mlp = torch.load(mlp_path)

    X = ['Video']

    mm_backbone_mlp_model, processor = get_processor(X, args, 'cuda', pretrained_checkpoint_mlp, model_path = 'LanguageBind/Video-LLaVA-7B')
    video_processor = processor['video']

    linear_proj = mm_backbone_mlp_model.mm_projector

    # 1. Load the pretrained weights
    pretrained_llm_checkpoint = lazy_load(pretrained_llm_path)
    # 2. Load the fine-tuned LoRA weights
    lora_checkpoint = lazy_load(lora_path)
    # 3. merge the two checkpoints
    model_state_dict = {**pretrained_llm_checkpoint, **lora_checkpoint}
    model.load_state_dict(model_state_dict, strict=True)
    print('Load llm base model from', pretrained_llm_path)
    print('Load lora model from', lora_path)

    # load mlp again, to en sure, not neccessary actually 
    linear_proj.load_state_dict(pretrained_checkpoint_mlp)
    linear_proj = linear_proj.cuda()
    print('Load mlp model again from', mlp_path)


    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)
    linear_proj.eval()
    

    tokenizer = Tokenizer(tokenizer_llm_path)
    print('Load tokenizer from', tokenizer_llm_path)


    while True:

        input_video_path = input("\033[0;34;40m Input video path: \033[0m")
        video_tensor = video_processor(input_video_path, return_tensors='pt')['pixel_values']

        if type(video_tensor) is list:
            tensor = [video.to('cuda', dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to('cuda', dtype=torch.float16) # (1,3,8,224,224)

        X_modalities = [tensor,['video']]

        video_feature = mm_backbone_mlp_model.get_multimodal_embeddings(X_modalities)

        prompt = input("\033[0;34;40m Your question: \033[0m")
        sample = {"instruction": prompt, "input": input_video_path}
        
        prefix = generate_prompt_mlp(sample)
        pre = torch.cat((tokenizer.encode(prefix.split('INPUT_VIDEO: ')[0] + "\n", bos=True, eos=False, device=model.device).view(1, -1), tokenizer.encode("INPUT_VIDEO: ", bos=False, eos=False, device=model.device).view(1, -1)), dim=1)

        prompt = (pre, ". ASSISTANT: ")
        encoded = (prompt[0], video_feature[0], tokenizer.encode(prompt[1], bos=False, eos=False, device=model.device).view(1, -1))

        
        t0 = time.perf_counter()
            
        output_seq = generate(
            model,
            idx=encoded,
            max_seq_length=4096,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
            tokenizer = tokenizer,
        )
        outputfull = tokenizer.decode(output_seq)
        output = outputfull.split("ASSISTANT:")[-1].strip()
        print("================================")
        print("Model output", output)
        print("================================")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
