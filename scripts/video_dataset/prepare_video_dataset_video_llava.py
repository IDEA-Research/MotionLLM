"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import os
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
sys.path.append(os.getcwd())
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np

from options import option

IGNORE_INDEX = -1

def prepare(
    destination_path: Path = Path("./data"), 
    tokenizer_path: Path = Path("./checkpoints/lit-llama/tokenizer.model"),
    max_seq_length: int = 2560,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    split: str = "train"
):
    """Prepare the Alpaca dataset for instruction tuning.
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    destination_path.mkdir(parents=True, exist_ok=True)

    file_path = f'/comp_robot/lushunlin/MotionGPT/data/video_dataset/video_llava_{split}.json'

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from?
    tokenizer = Tokenizer(tokenizer_path)

    with open(file_path, "r") as file:
        data = json.load(file)
    data_set = list(data)

    print(f"{split} set has {len(data_set):,} samples")

    print(f"Processing {split} split ...")
    data_set_new = []
    for sample in tqdm(data_set):
        # try:
        data_set_new.append(prepare_sample(sample, tokenizer, max_seq_length, mask_inputs))
            # import pdb; pdb.set_trace()

    data_set = data_set_new

    save_pt = f'/comp_robot/lushunlin/MotionGPT/data/video_dataset/video_llava_{split}.pt'
    torch.save(data_set, save_pt)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    """Processes a single sample.
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).


    """
    # import pdb; pdb.set_trace()
    # full_prompt = generate_prompt(example)
    # import pdb; pdb.set_trace()
    full_prompt = generate_prompt_mlp(example)
    full_prompt_and_response = full_prompt + example['output']
    
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)
    
    # extendedQA = example['QA'][1:]
    # for qa_item in extendedQA:
    #     q, a = qa_item["Q"], qa_item["A"]
    #     new_concat = "USER: " + q + "ASSISTANT: " + a
    #     full_prompt_and_response = full_prompt_and_response + new_concat
    #     encoded_new_concat = tokenize(tokenizer, new_concat, eos=True, max_length=max_length)
    #     encoded_full_prompt_and_response = torch.cat((encoded_full_prompt_and_response, encoded_new_concat))
        

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    # import pdb; pdb.set_trace()
    
    return {**example, "sys_command": generate_system_command(), "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)

def detokenizer(tokenizer: Tokenizer, tensor: torch.Tensor):
    '''
    tokenizer.decode(torch.tensor([13866,   338]))
    '''
    return tokenizer.decode(tensor)


def generate_prompt_mlp(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    # import pdb; pdb.set_trace()
    # try: 
    #     x = f"A chat between a curious user and an artificial intelligence assistant, paired with an input that provides further context. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {example['QA'][0]['Q']} INPUT_MOTION_TOKENS: {example['input']}. \nASSISTANT: " 
    # except:
    #     import pdb; pdb.set_trace()
    if example["input"]:
        return (
            f"A chat between a curious user and an artificial intelligence assistant, paired with an input that provides further context. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {example['instruction']} INPUT_VIDEO: {example['input']}. \nASSISTANT: "
        )
    return (
        f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {example['instruction']} ASSISTANT: "
    )
    
    # return (
    #     "Below is an instruction that describes a task, paired with an input that provides further context. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     f"### Instruction:\n{example['instruction']}\n\n### Input:\n", "\n\n### Response:"
    # )

def generate_prompt_mlp_mv_bench(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    # import pdb; pdb.set_trace()
    # try: 
    #     x = f"A chat between a curious user and an artificial intelligence assistant, paired with an input that provides further context. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {example['QA'][0]['Q']} INPUT_MOTION_TOKENS: {example['input']}. \nASSISTANT: " 
    # except:
    #     import pdb; pdb.set_trace()
    if example["input"]:
        return (
            f"A chat between a curious user and an artificial intelligence assistant, paired with an input that provides further context. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {example['instruction']} INPUT_VIDEO: {example['input']}. \nASSISTANT: "
        )
    return (
        f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {example['instruction']} ASSISTANT: "
    )
    
    # return (
    #     "Below is an instruction that describes a task, paired with an input that provides further context. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     f"### Instruction:\n{example['instruction']}\n\n### Input:\n", "\n\n### Response:"
    # )


def generate_system_command():
    return (
        f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    )
    

def main():
    args = option.get_args_parser()
    # prepare(split='train')
    # prepare(split='val')
    prepare(split='train_filter_wrong_decord_videos')
    prepare(split='val_filter_wrong_decord_videos')


if __name__ == "__main__":
    main()
