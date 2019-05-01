# PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields
PyTorch implementation of the latest Dec 2018 paper: 
https://arxiv.org/abs/1812.08008

<br/>
<b>Model Results:</b>

<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/model-results/10.png" align="center" width="100%"/>


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/model-results/12.png" align="center" width="100%"/>

<a href="https://www.youtube.com/watch?v=I7SoJzRPCPY">demo video</a>

<b>Joint Matching:</b> (With the help of predicted Part Affinity Field vectors)


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/matching-res.png" align="center" width="100%"/>


<b>Network Architecture:</b> (1st 10 layers from VGG-16 as backbone(F), 4 PAF stages(L), 2 Heatmap stages(S)) 


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/net-arch.png" align="center" width="100%"/>


<b>Training Image, Generated PAF's, Generated Joint Heatmaps:</b> 


<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/trn-pafs.png" align="center" width="100%"/>

<img src="https://raw.githubusercontent.com/DhruvJawalkar/PyTorch-OpenPose-Realtime-Multi-Person-2D-Pose-Estimation-using-Part-Affinity-Fields/master/trn-hms.png" align="center" width="100%"/>
