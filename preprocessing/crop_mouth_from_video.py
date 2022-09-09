import os
import glob
import argparse
from dataloader import AVSRDataLoader
from utils import save2npz


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--landmark-direc', default=None, help='landmark directory')
    parser.add_argument('--filename-path', default='./lrw500_detected_face.csv', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default=None, help='the directory of saving mouth ROIs')
    # -- convert to gray scale
    parser.add_argument('--convert-gray', default=False, action='store_true', help='convert2grayscale')
    # -- test set only
    parser.add_argument('--testset-only', default=False, action='store_true', help='process testing set only')

    args = parser.parse_args()
    return args

args = load_args()

dataloader = AVSRDataLoader(convert_gray=args.convert_gray)

modality = "video"

lines = open(args.filename_path).read().splitlines()
lines = list(filter(lambda x: 'test' == x.split('/')[-2], lines)) if args.testset_only else lines

for filename_idx, line in enumerate(lines):

    filename, person_id = line.split(',')
    print(f'idx: {filename_idx} \tProcessing.\t{filename}')

    video_filename = os.path.join(args.video_direc, filename+'.mp4')
    landmarks_filename = os.path.join(args.landmark_direc, filename+'.pkl')
    dst_filename = os.path.join( args.save_direc, filename+'.npz')

    assert os.path.isfile(video_filename), f"File does not exist. Path input: {video_filename}"
    assert os.path.isfile(landmarks_filename), f"File does not exist. Path input: {landmarks_filename}"

    if os.path.exists(dst_filename):
        continue

    # Extract mouth patches from segments
    sequence = dataloader.load_data(
        modality,
        video_filename,
        landmarks_filename,
    )

    try:
        if not os.path.exists(dst_filename):
            save2npz(dst_filename, data=sequence)
    except AssertionError:
        continue
