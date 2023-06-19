import torch
import numpy as np
import cv2

#from LoFTR.src.loftr import LoFTR, default_cfg
#from SuperGlue.models.utils import read_image
#from SuperGlue.models.matching import Matching

torch.set_grad_enabled(False)
from roma import roma_indoor, roma_outdoor

class RoMa_Matcher:
    def __init__(self, resize, outdoor = False, device = "cuda"):
        self.resize = resize[::-1]# w,h -> h,w
        self.roma_model = roma_outdoor(device = device) if outdoor else roma_indoor(device = device)
        self.roma_model.upsample_res = (864, 864)
        self.roma_model.h_resized = 560
        self.roma_model.w_resized = 560
        
    def match(self, pair_path):
        '''retrurn correspondences between images (w/ path pair_path)'''
        input_path0, input_path1 = pair_path
        warp, certainty = self.roma_model.match(input_path0, input_path1)
        matches, sparse_certainty = self.roma_model.sample(warp, certainty)
        kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, *self.resize, *self.resize)
        return np.concatenate((kpts1.cpu().numpy(),kpts2.cpu().numpy()),axis = 1)