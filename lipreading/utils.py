import os
import cv2
import json
import numpy as np


# -- Media utils
def extract_opencv(filename, bgr=False):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video if bgr else video[..., ::-1]


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content


def load_json( json_fp ):
    with open( json_fp, 'r' ) as f:
        json_content = json.load(f)
    return json_content


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)
