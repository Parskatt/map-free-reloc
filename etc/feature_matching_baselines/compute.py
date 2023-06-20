import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils import parse_7scenes_matching_pairs, parse_mapfree_query_frames, stack_pts, load_scannet_imgpaths
from matchers import RoMa_Matcher

MATCHERS = {'RoMa': RoMa_Matcher}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, default='Mapfree',
                        choices=['Scannet', '7Scenes', 'Mapfree'])
    parser.add_argument('--matcher', '-m', type=str, default='RoMa',
                        choices=MATCHERS.keys())
    parser.add_argument('--scenes', '-sc', type=str, nargs='*', default=None)
    parser.add_argument('--outdoor', action='store_true',
                        help='use outdoor SG/LoFTR model. If not specified, use indoor models')
    args = parser.parse_args()

    args.data_root = Path('data/map-free/')
    test_scenes = [folder for folder in (args.data_root / 'test').iterdir() if folder.is_dir()]
    args.scenes = test_scenes
    resize = 540, 720

    return args, MATCHERS[args.matcher](resize, args.outdoor)


if __name__ == '__main__':
    args, matcher = get_parser()
    for scene_dir in args.scenes:
        if os.path.exists(scene_dir / f'correspondences_{args.matcher}.npz'):
            continue
        query_frames_paths = parse_mapfree_query_frames(scene_dir / 'poses.txt')
        im_pairs_path = [(str(scene_dir / 'seq0' / 'frame_00000.jpg'),
                            str(scene_dir / qpath)) for qpath in query_frames_paths]

        pts_stack = list()
        print(f'Started {scene_dir.name}')
        for pair in tqdm(im_pairs_path[::5]):
            pts = matcher.match(pair)
            pts_stack.append(pts)
        pts_stack = stack_pts(pts_stack)
        results = {'correspondences': pts_stack}
        np.savez_compressed(scene_dir / f'correspondences_{args.matcher}.npz', **results)
        print(f'Finished {scene_dir.name}')