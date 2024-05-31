import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--prompt', type=str, default="Generate a textual description corresponding to the given sequence of human motion tokens.", help='task description')
    parser.add_argument('--input', type=str, help='generation condictions')
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--pretrained_llama', type=str, default="13B")
    parser.add_argument('--out_dir', type=str, default='./out/', help='output directory')
    parser.add_argument('--vqvae_pth', type=str, default='/comp_robot/lushunlin/MotionGPT/checkpoints/pretrained_vqvae/t2m.pth', help='path to the pretrained vqvae pth')
    parser.add_argument('--resume_pth', type=str, help='path to saved finetuned model')
    parser.add_argument('--lora_path', type=str, help='path to fintuned model for evaluation')
    parser.add_argument('--mlp_path', type=str, help='mlp path')
    parser.add_argument('--data_dir', type=str, default='./data/', help='dataset directory')


    ## lora
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    ## llama
    parser.add_argument('--block_size', type=int, default=512)

    ## train
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    # parser.add_argument('--learning_rate', type=float, default=3e-3, help='learning rate')
    parser.add_argument('--learning_rate_lora', type=float, default=3e-3, help='learning rate of lora')
    parser.add_argument('--learning_rate_mlp', type=float, default=3e-3, help='learning rate of mlp')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100, help='warmup steps')
    parser.add_argument('--eval_interval', type=int, default=100, help='evaluation frequency')
    parser.add_argument('--save_interval', type=int, default=100, help='model save frequency')
    parser.add_argument('--eval_iters', type=int, default=100, help='number of evaluation ierations')
    parser.add_argument('--log_interval', type=int, default=1, help='log frequency')

    ## vqvae
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing vqvae training.')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## visualization
    parser.add_argument("--render", action='store_true', help='render smpl')
    parser.add_argument("--motion_vq_token_path", type=str, help='vq token path for motion visualization')


    ## for motionx zero shot
    parser.add_argument('--motionx_zero_shot_path', type=str, help='zero shot motion dataset directory')

    parser.add_argument("--projectionnn", action='store_true', help='MLP projection')
    parser.add_argument("--diverse", action='store_true', help='diverse description')
    parser.add_argument("--vinilla", action='store_true', help='vinilla motion')
    

    # for video llava
    parser.add_argument('--image_tower', type=str, default='LanguageBind/LanguageBind_Image', help='if use multimodal image tower')
    parser.add_argument('--video_tower', type=str, default='LanguageBind/LanguageBind_Video_merge', help='if use multimodal video tower')
    parser.add_argument('--mm_vision_select_layer', type=int, default=-2, help='if use multimodal video tower')
    parser.add_argument('--mm_projector_type', type=str, default='mlp2x_gelu', help='if use multimodal video tower')
    parser.add_argument('--mm_hidden_size', type=int, default=1024, help='if use multimodal video tower')
    parser.add_argument('--hidden_size', type=int, default=4096, help='if use multimodal video tower')

    # for mvbench save
    parser.add_argument('--model_type', type=str, default=None, help='if use multimodal video tower')

    return parser.parse_args()
