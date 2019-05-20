import torch
import torchvision
import torch.nn as nn
from torchvision.models import vgg19
import model_utils

class F(nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:23]
        self.conv_4_3_and_4_4 = nn.Sequential(
                                              nn.Conv2d(512, 256, 3, 1, 1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(256, 125, 3, 1, 1),
                                              nn.BatchNorm2d(125),
                                              nn.ReLU(inplace=True)
                                              )
        model_utils.freeze_all_layers(self.vgg)
    
    def freeze_all_layers(self):
        model_utils.freeze_all_layers(self.vgg)
        model_utils.freeze_all_layers(self.conv_4_3_and_4_4)
    
    def forward(self, x):
        return self.conv_4_3_and_4_4(self.vgg(x))

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        #64, 32, 32
        super(Conv_Block, self).__init__()
        self.C1 = nn.Sequential(
                                nn.Conv2d(in_channels, 64, 3, 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True)
                                )
        self.C2 = nn.Sequential(
                                nn.Conv2d(64, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True)
                                )
        self.C3 = nn.Sequential(
                                nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True)
                                )

    def forward(self, x):
        c1_out = self.C1(x)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        return torch.cat((c1_out, c2_out, c3_out), dim=1)


def get_stage_block(in_channels, out_channels):
    return nn.Sequential(
                         Conv_Block(in_channels),
                         Conv_Block(128),
                         Conv_Block(128),
                         Conv_Block(128),
                         Conv_Block(128),
                         nn.Conv2d(128,128,1,1,0),
                         nn.BatchNorm2d(128),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(128,out_channels,1,1,0)
                         )

class PAF_Stages(nn.Module):
    def __init__(self, in_channels=128, paf_out_channels=38+8, in_training=False):
        super(PAF_Stages, self).__init__()
        self.Stage1 = get_stage_block(in_channels, paf_out_channels)
        self.Stage2 = get_stage_block(in_channels+paf_out_channels, paf_out_channels)
        self.Stage3 = get_stage_block(in_channels+paf_out_channels, paf_out_channels)
        self.Stage4 = get_stage_block(in_channels+paf_out_channels, paf_out_channels)
        self.in_training = in_training
        self.current_training_stage = -1
    
    def set_current_training_stage(self, stage):
        self.in_training = True
        self.current_training_stage = stage
        model_utils.freeze_other_paf_stages(self, stage)
    
    def unfreeze_all_stages(self):
        model_utils.unfreeze_all_paf_stages(self)
    
    def freeze_all_stages(self):
        model_utils.freeze_all_paf_stages(self)
    
    def set_to_training(self):
        self.in_training = True

    def set_to_inference(self):
        self.in_training = False

    def forward(self, im_46x46, F):
        if(self.in_training):
            res = []
            if(self.current_training_stage>=1):
                o1 = self.Stage1(torch.cat((im_46x46.clone(), F.clone()), dim=1))
                res.append(o1)
            if(self.current_training_stage>=2):
                o2 = self.Stage2(torch.cat((im_46x46.clone(), F.clone(), o1), dim=1))
                res.append(o2)
            if(self.current_training_stage>=3):
                o3 = self.Stage3(torch.cat((im_46x46.clone(), F.clone(), o2), dim=1))
                res.append(o3)
            if(self.current_training_stage>=4):
                o4 = self.Stage4(torch.cat((im_46x46.clone(), F.clone(), o3), dim=1))
                res.append(o4)
            return res
        else:
            o1 = self.Stage1(torch.cat((im_46x46.clone(), F.clone()), dim=1))
            o2 = self.Stage2(torch.cat((im_46x46.clone(), F.clone(), o1), dim=1))
            o3 = self.Stage3(torch.cat((im_46x46.clone(), F.clone(), o2), dim=1))
            o4 = self.Stage4(torch.cat((im_46x46.clone(), F.clone(), o3), dim=1))
            return o4

class Heatmap_Stages(nn.Module):
    def __init__(self, in_channels=128+38+8, hm_out_channels=17+1, in_training=False):
        super(Heatmap_Stages, self).__init__()
        self.Stage1 = get_stage_block(in_channels, hm_out_channels)
        self.Stage2 = get_stage_block(in_channels+hm_out_channels, hm_out_channels)
        self.in_training = in_training
        self.current_training_stage = -1
    
    def set_current_training_stage(self, stage):
        self.in_training = True
        self.current_training_stage = stage
        model_utils.freeze_other_hm_stages(self, stage)
    
    def unfreeze_all_stages(self):
        model_utils.unfreeze_all_hm_stages(self)
    
    def freeze_all_stages(self):
        model_utils.freeze_all_hm_stages(self)
    
    def set_to_training(self):
        self.in_training = True

    def set_to_inference(self):
        self.in_training = False
    
    def forward(self, im_46x46, F, L):
        if(self.in_training):
            res = []
            if(self.current_training_stage>=1):
                o1 = self.Stage1(torch.cat((im_46x46.clone(), F.clone(), L.clone()), dim=1))
                res.append(o1)
            if(self.current_training_stage>=2):
                o2 = self.Stage2(torch.cat((im_46x46.clone(), F.clone(), L.clone(), o1), dim=1))
                res.append(o2)
            return res
        else:
            o1 = self.Stage1(torch.cat((im_46x46.clone(), F.clone(), L.clone()), dim=1))
            o2 = self.Stage2(torch.cat((im_46x46.clone(), F.clone(), L.clone(), o1), dim=1))
            return o2

class Net(nn.Module):
    def __init__(self, in_training=False):
        super(Net, self).__init__()
        self.in_training = in_training
        self.F = F()
        self.PAF_Stages = PAF_Stages(in_training=self.in_training)
        self.Heatmap_Stages = Heatmap_Stages(in_training=self.in_training)
        self.train_heatmaps = False
        if(self.in_training):
            self.PAF_Stages.set_current_training_stage(1)
            self.Heatmap_Stages.freeze_all_stages()

    def set_to_inference(self):
        self.in_training = False
        self.F.freeze_all_layers()
        self.PAF_Stages.set_to_inference()
        self.Heatmap_Stages.set_to_inference()
    
    def freeze_F(self):
        model_utils.freeze_all_layers(self.F)
    
    def unfreeze_F(self):
        model_utils.unfreeze_all_layers(self.F)
    
    def train_paf_stage(self, stg):
        self.PAF_Stages.set_current_training_stage(stg)
    
    def freeze_all_paf_stages(self):
        self.PAF_Stages.freeze_all_stages()
        self.PAF_Stages.set_to_inference()
    
    def unfreeze_all_paf_stages(self):
        self.PAF_Stages.unfreeze_all_stages()
    
    def train_hm_stage(self, stg):
        self.train_heatmaps = True
        self.Heatmap_Stages.set_current_training_stage(stg)

    def freeze_all_hm_stages(self):
        self.Heatmap_Stages.freeze_all_stages()
    
    def unfreeze_all_hm_stages(self):
        self.Heatmap_Stages.unfreeze_all_stages()
    
    def set_train_heatmaps(flag):
        self.train_heatmaps = flag
    
    def forward(self, img, im_46x46):
        image_features = self.F(img)
        if(self.in_training):
            pafs_op = self.PAF_Stages(im_46x46, image_features)
            #heatmaps_op = []
            #if(self.train_heatmaps):
            #    heatmaps_op = self.Heatmap_Stages(im_46x46, image_features, pafs_op)
            heatmaps_op = self.Heatmap_Stages(im_46x46, image_features, pafs_op[3])
            return pafs_op, heatmaps_op
        else:
            pafs = self.PAF_Stages(im_46x46, image_features)
            heatmaps = self.Heatmap_Stages(im_46x46, image_features, pafs)
            return pafs, heatmaps
