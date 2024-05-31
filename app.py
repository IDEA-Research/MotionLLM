import shutil
import subprocess

import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
from decord import VideoReader, cpu
import uvicorn
from transformers import TextStreamer

import hashlib
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional
from typing import Dict, List, Literal, Optional, Tuple
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable

import lightning as L
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from generate import generate as generate_
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


title_markdown = ("""<div class="embed_hidden" style="text-align: center;">
    <h1>MotionLLM: Understanding Human Behaviors from Human Motions and Videos</h1>
    <h3>
        <a href="https://lhchen.top" target="_blank" rel="noopener noreferrer">Ling-Hao Chen</a><sup>😎 1, 3</sup>,
        <a href="https://shunlinlu.github.io" target="_blank" rel="noopener noreferrer">Shunlin Lu</a><sup>😎 2, 3</sup>,
        <br>
        <a href="https://ailingzeng.sit" target="_blank" rel="noopener noreferrer">Ailing Zeng</a><sup>3</sup>,
        <a href="https://haozhang534.github.io/" target="_blank" rel="noopener noreferrer">Hao Zhang</a><sup>3, 4</sup>,
        <a href="https://wabyking.github.io/old.html" target="_blank" rel="noopener noreferrer">Benyou Wang</a><sup>2</sup>,
        <a href="http://zhangruimao.site" target="_blank" rel="noopener noreferrer">Ruimao Zhang</a><sup>2</sup>,
        <a href="https://leizhang.org" target="_blank" rel="noopener noreferrer">Lei Zhang</a><sup>🤗 3</sup>
    </h3>
    <h3><sup>😎</sup><i>Co-first author. Listing order is random.</i> &emsp; <sup>🤗</sup><i>Corresponding author.</i></h3>
    <h3>
        <sup>1</sup>THU &emsp;
        <sup>2</sup>CUHK (SZ) &emsp;
        <sup>3</sup>IDEA Research  &emsp;
        <sup>4</sup>HKUST
    </h3>
</div>
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <img src="https://lhchen.top/MotionLLM/assets/img/highlight.png" alt="MotionLLM" style="width:60%; height: auto; align-items: center;">
</div>

""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""


tos_markdown = ("""
*We are now working to support the motion branch of the MotionLLM model.

### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. 
It is forbidden to use the service to generate content that is illegal, harmful, violent, racist, or sexual
The usage of this service is subject to the IDEA License.
""")


learn_more_markdown = ("""
### License
License for Non-commercial Scientific Research Purposes

IDEA grants you a non-exclusive, worldwide, non-transferable, non-sublicensable, revocable, royalty free and limited license under IDEA’s copyright interests to reproduce, distribute, and create derivative works of the text, videos, codes solely for your non-commercial research purposes.

Any other use, in particular any use for commercial, pornographic, military, or surveillance, purposes is prohibited.  

Text and visualization results are owned by International Digital Economy Academy (IDEA).

You also need to obey the original license of the dependency models/data used in this service.
""")



class LlavaMetaModel:

    def __init__(self, config, pretrained_checkpoint):
        super(LlavaMetaModel, self).__init__()
        # import pdb; pdb.set_trace()
        if hasattr(config, "mm_image_tower") or hasattr(config, "image_tower"):
            self.image_tower = build_image_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_video_tower") or hasattr(config, "video_tower"):
            self.video_tower = build_video_tower(config, delay_load=True)
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
        # import pdb; pdb.set_trace()
        # videos: torch.Size([1, 3, 8, 224, 224])
        video_features = self.get_video_tower()(videos) # torch.Size([1, 2048, 1024])
        video_features = self.mm_projector(video_features.float()) # torch.Size([1, 2048, 4096])
        return video_features
    
    def get_multimodal_embeddings(self, X_modalities):
        Xs, keys= X_modalities
        
        X_features = getattr(self, f'encode_{keys[0]}s')(Xs)  # expand to get batchsize

        return X_features


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


class Conversation():
    def __init__(self, output=None, input_prompt=None, prompt=None):
        if output is None:
            self.messages = []
        else:
            self.messages = []
            self.append_message(prompt, input_prompt, output)

    def append_message(self, output, input_prompt, prompt, show_images):
        # print(output)
        # print(input_prompt)
        # print(prompt)
        # print(show_images)
        self.messages.append((output, input_prompt, prompt, show_images))

    def to_gradio_chatbot(self, show_images=None, output_text=None):
        # return a list
        if show_images is None:
            show_images = self.messages[-1][3]
            output_text = self.messages[-1][0]
        return [
            [show_images, output_text]
        ]
        
    def get_info(self):
        return self.messages[-1][0], self.messages[-1][1]


class ConversationBuffer():
    def __init__(self, input_text):
        self.buffer_ = []
        self.buffer.append(input_text)


def init_conv():
    conv = Conversation()
    return conv


def get_processor(X, config, device, pretrained_checkpoint_tower, model_path = 'LanguageBind/MotionLLM-7B'):
    mm_backbone_mlp_model = LlavaMetaModel(config, pretrained_checkpoint_tower)

    processor = {} 
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


def motionllm(
    args, 
    input_video_path: str,
    text_en_in: str, 
    quantize: Optional[str] = None,
    dtype: str = "float32",
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.8,
    accelerator: str = "auto",):
    
    video_tensor = video_processor(input_video_path, return_tensors='pt')['pixel_values']

    if type(video_tensor) is list:
        tensor = [video.to('cuda', dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to('cuda', dtype=torch.float16) # (1,3,8,224,224)

    X_modalities = [tensor,['video']]
    video_feature = mm_backbone_mlp_model.get_multimodal_embeddings(X_modalities)
    prompt = text_en_in
    input_prompt = prompt
    
    sample = {"instruction": prompt, "input": input_video_path}
    
    prefix = generate_prompt_mlp(sample)
    pre = torch.cat((tokenizer.encode(prefix.split('INPUT_VIDEO: ')[0] + "\n", bos=True, eos=False, device=model.device).view(1, -1), tokenizer.encode("INPUT_VIDEO: ", bos=False, eos=False, device=model.device).view(1, -1)), dim=1)

    prompt = (pre, ". ASSISTANT: ")
    encoded = (prompt[0], video_feature[0], tokenizer.encode(prompt[1], bos=False, eos=False, device=model.device).view(1, -1))
    
    t0 = time.perf_counter()
        
    output_seq = generate_(
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
    print(output)
    print("================================")
    
    return output, input_prompt, prompt


def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    # print(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename


def generate(image1, video, textbox_in, first_run, state, images_tensor):
    flag = 1

    image1 = image1 if image1 else "none"
    video = video if video else "none"
    
    if type(state) is not Conversation:
        state = init_conv()
        images_tensor = [[], []]

    first_run = False if len(state.messages) > 0 else True
    text_en_in = textbox_in.replace("picture", "image")
    output, input_prompt, prompt = motionllm(args, video, text_en_in)

    text_en_out = output
    textbox_out = text_en_out

    show_images = ""
    if os.path.exists(image1):
        filename = save_image_to_local(image1)
        show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'

    if os.path.exists(video):
        filename = save_video_to_local(video)
        show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

    show_images = textbox_in + "\n" + show_images
    state.append_message(output, input_prompt, prompt, show_images)

    torch.cuda.empty_cache()
    
    return (state, state.to_gradio_chatbot(show_images, output), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=image1 if os.path.exists(image1) else None, interactive=True), gr.update(value=video if os.path.exists(video) else None, interactive=True))

def regenerate(state):
    if len(state.messages) > 0:
        tobot = state.to_gradio_chatbot()
        tobot[-1][1] = None 
        textbox = state.messages[-1][1]
        state.messages.pop(-1)
        return state, tobot, False, textbox
    return (state, [], True)


def clear_history(state):
    state = init_conv()
    try:
        tgt = state.to_gradio_chatbot()
    except:
        tgt = [None, None]
    return (gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        True, state, tgt, [[], []])


def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def logging_up(video, state):
    try:
        state.get_info()
    except:
        return False
    action = "upvote"
    # Get the current time
    current_time = str(time.time())

    # Create an md5 object
    hash_object = hashlib.md5(current_time.encode())

    # Get the hexadecimal representation of the hash
    md5_hash = get_md5(video) + "-" + hash_object.hexdigest()
    
    command = f"cp {video} ./feedback/{action}/mp4/{md5_hash}.mp4"
    os.system(command)
    with open (f"./feedback/{action}/txt/{md5_hash}.txt", "w") as f:
        out, prp = state.get_info()
        f.write(f"==========\nPrompt: {prp}\n==========\nOutput: {out}==========\n")
    return True


def logging_down(video, state):
    try:
        state.get_info()
    except:
        return False
    action = "downvote"
    # Get the current time
    current_time = str(time.time())

    # Create an md5 object
    hash_object = hashlib.md5(current_time.encode())

    # Get the hexadecimal representation of the hash
    md5_hash = get_md5(video) + "-" + hash_object.hexdigest()
    
    command = f"cp {video} ./feedback/{action}/mp4/{md5_hash}.mp4"
    os.system(command)
    with open (f"./feedback/{action}/txt/{md5_hash}.txt", "w") as f:
        out, prp = state.get_info()
        f.write(f"==========\nPrompt: {prp}\n==========\nOutput: {out}==========\n")
    return True


torch.set_float32_matmul_precision("high")
warnings.filterwarnings('ignore')
args = option.get_args_parser()

conv_mode = "llava_v1"
model_path = 'LanguageBind/Video-LLaVA-7B'
device = 'cuda'
load_8bit = False
load_4bit = True
dtype = torch.float16

if not os.path.exists("temp"):
    os.makedirs("temp")

lora_path = Path(args.lora_path)
pretrained_llm_path = Path(f"./checkpoints/vicuna-7b-v1.5/lit_model.pth")
tokenizer_llm_path = Path("./checkpoints/vicuna-7b-v1.5/tokenizer.model")

# assert lora_path.is_file()
assert pretrained_llm_path.is_file()
assert tokenizer_llm_path.is_file()

accelerator = "auto"
fabric = L.Fabric(accelerator=accelerator, devices=1)

dtype = "float32"
dt = getattr(torch, dtype, None)
if not isinstance(dt, torch.dtype):
    raise ValueError(f"{dtype} is not a valid dtype.")
dtype = dt

quantize = None
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

print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())


app = FastAPI()

textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )

with gr.Blocks(title='MotionLLM', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(title_markdown)
    state = gr.State()
    buffer_ = gr.State()
    first_run = gr.State()
    images_tensor = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            image1 = gr.State()
            video = gr.Video(label="Input Video")

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/Play_Electric_guitar_16_clip1.mp4",
                        "why is the girl so happy",
                    ],
                    [
                        f"{cur_dir}/examples/guoyoucai.mov",
                        "what is the feeling of him",
                    ],
                    [
                        f"{cur_dir}/examples/sprint_run_18_clip1.mp4",
                        "Why is the man running so fast?",
                    ],
                    [
                        f"{cur_dir}/examples/lift_weight.mp4",
                        "Assume you are a fitness coach, refer to the video of the professional athlete, please analyze specific action essentials in steps and give detailed instruction.",
                    ],
                    [
                        f"{cur_dir}/examples/Shaolin_Kung_Fu_Wushu_Selfdefense_Sword_Form_Session_22_clip3.mp4",
                        "wow, can you teach me the motion, step by step in detail",
                    ],
                    [
                        f"{cur_dir}/examples/mabaoguo.mp4",
                        "why is the video funny?",
                    ],
                    [
                        f"{cur_dir}/examples/COBRA_PUSH_UPS_clip2.mp4",
                        "describe the body movement of the woman",
                    ],
                    [
                        f"{cur_dir}/examples/sample_demo_1.mp4",
                        "Why is this video interesting?",
                    ],
                ],
                inputs=[video, textbox],
            )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="MotionLLM", bubble_full_width=True).style(height=875)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=True
                    )
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)
    
    tmp = gr.State()
    upvote_btn.click(logging_up, [video, state], [tmp])
    
    downvote_btn.click(logging_down, [video, state], [tmp])

    submit_btn.click(generate, [image1, video, textbox, first_run, state, images_tensor],
                     [state, chatbot, first_run, textbox, images_tensor, image1, video])

    regenerate_btn.click(regenerate, [state], [state, chatbot, first_run, textbox]).then(
        generate, [image1, video, textbox, first_run, state, images_tensor], [state, chatbot, first_run, textbox, images_tensor, image1, video])

    clear_btn.click(clear_history, [state],
                    [image1, video, textbox, first_run, state, chatbot, images_tensor])

app = gr.mount_gradio_app(app, demo, path="/")
uvicorn.run(app, host="0.0.0.0", port=6657)