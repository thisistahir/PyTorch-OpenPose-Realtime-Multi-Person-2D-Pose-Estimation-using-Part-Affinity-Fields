import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_utils
from CONSTANTS import *

def plot_matching_results(img, part_matches_map, all_joint_detections_map, pred_pafs):
    C,S = ['g','r','y','w','c','m','k'], ['o','+','h','.','s','P','p','*','H','x','D','d','8']
    
    fig, axes = plt.subplots(1,19, figsize=(70,5))
    for i, ax in enumerate(axes.flat): 
        ax.axis('off')
        part_pair = SKELETON[i]
        part_pair_tuple = (part_pair[0], part_pair[1])
        
        if(len(part_matches_map[part_pair_tuple])):
            ax.text(10,10, keypoint_labels[part_pair[0]]+'->'+keypoint_labels[part_pair[1]], va='top', color="white", fontsize=12)
            ax.imshow(img)

            px, py = pred_pafs[(2*i)].numpy(), pred_pafs[(2*i)+1].numpy()
            mask = np.logical_or(px, py)
            ax.imshow(mask, 'jet', interpolation='none', alpha=0.5)


            for matched_pt_pair in part_matches_map[part_pair_tuple]:
                pts = np.array([matched_pt_pair[0], matched_pt_pair[1]])
                ax.plot(pts[:,0], pts[:,1], C[np.random.randint(0,len(C))]+S[np.random.randint(0,len(S))], markersize=12) 
                ax.plot(pts[:,0], pts[:,1], 'w-', linewidth=2)
                ax.text(pts[0,0], pts[0,1], round(matched_pt_pair[2], 2), color='white')

            detected_parts_1 = all_joint_detections_map[part_pair[0]]
            detected_parts_2 = all_joint_detections_map[part_pair[1]]
            ax.plot(detected_parts_1[:,0], detected_parts_1[:,1], 'w+')
            ax.plot(detected_parts_2[:,0], detected_parts_2[:,1], 'w+')
        else:  
            ax.figsize=(0,0)

    plt.tight_layout()
    
def plot_heatmaps(img, masks, idx_to_keypoint_type=idx_to_keypoint_type, figsize=(16,12)):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        ax.axis('off')
        if(i<17):
            joint_type = idx_to_keypoint_type[i]
            peaks = model_utils.get_peaks(masks[i])
            ax.text(10,10, joint_type, va='top', color="white", fontsize=12)
            ax.imshow(img)
            ax.imshow(masks[i], 'jet', interpolation='none', alpha=0.5)
            ax.plot(peaks[:,0], peaks[:,1], 'w+')
        if(i==17):
            joint_type = "background"
            ax.text(10,10, joint_type, va='top', color="white", fontsize=12)
            ax.imshow(img)
            ax.imshow(masks[i], 'jet', interpolation='none', alpha=0.5)

    plt.tight_layout()

def plot_pafs(img, pafs, joint_pairs=part_pairs, figsize=(16,12)):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        ax.axis('off')
        if(i<19):
            ax.text(10,10, joint_pairs[i][0]+'->'+joint_pairs[i][1], va='top', color="white", fontsize=12)
            ax.imshow(img)
            mask = np.logical_or(pafs[2*i], pafs[(2*i) + 1]).astype(int)
            ax.imshow(mask, 'jet', interpolation='none', alpha=0.7)
    plt.tight_layout()

def plot_paf_maps_from_annotations(img, keypoints, joint_pairs=part_pairs, keypoint_type_to_idx=keypoint_type_to_idx, n_items=19, figsize=(16,12), limb_width=5):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        ax.axis('off')
        if(i<19):
            ax.text(10,10, joint_pairs[i][0]+'->'+joint_pairs[i][1], va='top', color="white", fontsize=12)
            joint_pair_paf,_ = model_utils.calculate_paf_mask(img, joint_pairs[i], keypoints, keypoint_type_to_idx, limb_width)
            ax.imshow(img)
            ax.imshow(joint_pair_paf.transpose(), 'jet', interpolation='none', alpha=0.5)
    plt.tight_layout()

def plot_heat_maps_from_annotations(img, anns, n_items=17, figsize=(16,12), sigma=7):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    img = np.array(img)
    fliped_img = img.transpose((1,0,2))
    kps = model_utils.get_keypoints_from_annotations(anns)
    
    for i,ax in enumerate(axes.flat):
        ax.axis('off')
        if(i<17):
            joint_type = idx_to_keypoint_type[i]
            ax.text(10,10, joint_type, va='top', color="white", fontsize=12)
            mask,_ = model_utils.calculate_heatmap(img, i, kps, sigma)
            ax.imshow(img)
            ax.imshow(mask.transpose(), 'jet', interpolation='none', alpha=0.5)
    plt.tight_layout()