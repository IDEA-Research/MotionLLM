import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('mm_image_tower', action='store_true', default=True, help='if use multimodal image tower')
    parser.add_argument('mm_video_tower', action='store_true', default=True, help='if use multimodal video tower')

    return parser.parse_args()
