# PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields

![output.gif](https://oregonstate.box.com/s/2s390ub7b5880o4qnrq3io5xdmxhjsxe) 

https://oregonstate.box.com/s/2s390ub7b5880o4qnrq3io5xdmxhjsxe

PyTorch implementation of the Dec 2018 paper: 
https://arxiv.org/abs/1812.08008

Go through estimate-pose.ipynb for training and evaluation code on sample image, video.

Model has been trained on MS-COCO 2014 dataset on 368x368 and 184x184 resolutions (model-wts-368.ckpt, model-wts-184.ckpt). 

Has additional PAF's from Shoulder->Wrist and Hip->Ankle for improved matching in crowded scenes.

You can change the threholds for PAF map o/p values, Heatmap threshold and part matching threshold in CONFIG.py (for more, less conf joint preds vs less, more confident).  

Part Matching formulation uses Munkres for one-one least cost matching. 

<br/>
<b>Model Results:</b>

<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/model-results/10.png" align="center" width="100%"/>


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/model-results/12.png" align="center" width="100%"/>


<b>Joint Matching:</b> (With the help of predicted Part Affinity Field vectors)


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/matching-res.png" align="center" width="100%"/>


<b>Network Architecture:</b> (1st 10 layers from VGG-16 as backbone(F), 4 PAF stages(L), 2 Heatmap stages(S)) 


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/net-arch.png" align="center" width="100%"/>


<b>Training Image, Generated PAF's, Generated Joint Heatmaps:</b> 


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/trn-pafs.png" align="center" width="100%"/>

<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/trn-hms.png" align="center" width="100%"/>
