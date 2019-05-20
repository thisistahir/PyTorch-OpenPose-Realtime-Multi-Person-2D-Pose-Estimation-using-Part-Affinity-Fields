import torch
from PIL import Image
import time
import model_utils
from model_utils import timeit
import numpy as np

class COCO_Person_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, processed_files, tfms, tensor_tfms, im_sz=368):
        super(COCO_Person_Dataset, self).__init__()
        self.image_dir = image_dir
        self.im_ids = np.load(processed_files["im_ids"])
        self.img_id_to_annotations = np.load(processed_files["img_id_to_annotations"], allow_pickle=True).ravel()[0]
        self.img_id_to_image_info = np.load(processed_files["img_id_to_image_info"], allow_pickle=True).ravel()[0]
        
        self.tfms = tfms
        self.tensor_tfms = tensor_tfms
        self.get_heatmap_masks = model_utils.get_heatmap_masks_optimized     
        self.get_paf_masks = model_utils.get_paf_masks_optimized
        self.limb_width = 5*(im_sz/368)
        self.sigma = 7*(im_sz/368)
        self.len = len(self.im_ids)
        self.heatmap_ps_map = model_utils.get_heatmap_ps_map(im_sz)
            
    #@timeit
    def __getitem__(self, index):
        im_id = self.im_ids[index]
        image = Image.open(self.image_dir+self.img_id_to_image_info[im_id]['file_name'])
        annotations = self.img_id_to_annotations[im_id]
        keypoints = model_utils.get_keypoints_from_annotations(annotations)
        
        if self.tfms:
            tfmd_sample = self.tfms({"image":image, "keypoints":keypoints})
            image, image_stg_input, keypoints = tfmd_sample["image"], tfmd_sample["image_stg_input"], tfmd_sample["keypoints"]
        
        heatmaps, HM_BINARY_IND = self.get_heatmap_masks(image, keypoints, self.sigma, self.heatmap_ps_map)
        pafs, PAF_BINARY_IND = self.get_paf_masks(image_stg_input, keypoints, limb_width=self.limb_width) 
            
        if self.tensor_tfms:
            res = self.tensor_tfms({"image":image, "image_stg_input": image_stg_input, "pafs":pafs, "PAF_BINARY_IND":PAF_BINARY_IND, "heatmaps":heatmaps, "HM_BINARY_IND":HM_BINARY_IND})
            image = res["image"]
            image_stg_input = res["image_stg_input"]
            pafs = res["pafs"]
            PAF_BINARY_IND = res["PAF_BINARY_IND"]
            heatmaps = res["heatmaps"]
            HM_BINARY_IND = res["HM_BINARY_IND"]
        return (image, image_stg_input, pafs, PAF_BINARY_IND, heatmaps, HM_BINARY_IND)
    
    def __len__(self):
        return self.len

