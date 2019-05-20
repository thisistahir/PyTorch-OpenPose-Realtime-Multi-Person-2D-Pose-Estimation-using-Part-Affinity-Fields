import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageOps

class Denorm(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor.mul(self.std).add(self.mean)
    
class RandomCrop(object):
    def __init__(self, size=368, p=1):
        self.sz = size
        self.p = p
        
    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']
        H, W = image.height, image.width
        
        if (W>self.sz and H>self.sz and np.random.random()>(1-self.p)):
            x = np.random.randint(W - self.sz)
            y = np.random.randint(H - self.sz)
            croped_img = image.crop(box=(x,y, x+self.sz, y+self.sz))
            
            keypoints[keypoints[:,:,0]<x] = np.array([0,0,0])
            keypoints[keypoints[:,:,1]<y] = np.array([0,0,0])
            keypoints[keypoints[:,:,0]>x+self.sz] = np.array([0,0,0])
            keypoints[keypoints[:,:,1]>y+self.sz] = np.array([0,0,0])
            keypoints[:,:,:2][keypoints[:,:,2]>0] = keypoints[:,:,:2][keypoints[:,:,2]>0] - np.array([[x, y]])
            
            return { 'image' : croped_img, 'keypoints':keypoints }
        else: return sample

class ResizeImgAndKeypoints(object):
    def __init__(self, size=368):
        self.size = size
        self.paf_sz = int(size*46/368)
        self.Resize = transforms.Resize((self.paf_sz, self.paf_sz))
    
    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints'].copy().astype(float) #2x17x3
        IM_H, IM_W = image.height, image.width
        if(IM_H > IM_W):
            w = int(self.size*IM_W/IM_H)
            h = self.size
            pad_val = int((self.size-w)/2)
            pad = (self.size-w-pad_val,0,pad_val ,0)
            keypoints[:,:,0] = keypoints[:,:,0]*(w/IM_W)
            keypoints[:,:,0][keypoints[:,:,2]>0] += self.size-w-pad_val
            keypoints[:,:,1] = keypoints[:,:,1]*(self.size/IM_H)
        
        else:
            h = int(self.size*IM_H/IM_W)
            w = self.size
            pad_val = int((self.size-h)/2)
            pad = (0,self.size-h-pad_val,0,pad_val)
            keypoints[:,:,0] = keypoints[:,:,0]*(self.size/IM_W)
            keypoints[:,:,1] = keypoints[:,:,1]*(h/IM_H)
            keypoints[:,:,1][keypoints[:,:,2]>0] += self.size-h-pad_val
        
        resized_img = ImageOps.expand(image.resize((w,h),resample=Image.BILINEAR), pad)
        return { 'image' : resized_img , 'image_stg_input': self.Resize(resized_img),'keypoints' : keypoints }

class FlipHR(object):
    def __init__(self, p=0.25):
        self.p = p
    
    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']
        
        if np.random.random() > (1-self.p):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = image.size
            keypoints[:, :, 0][keypoints[:, :, 2]>0] = w - keypoints[:, :, 0][keypoints[:, :, 2]>0]
            copy = keypoints.copy()
            keypoints[:,1,:], keypoints[:,2,:] = copy[:,2,:], copy[:,1,:]
            keypoints[:,3,:], keypoints[:,4,:] = copy[:,4,:], copy[:,3,:]
            keypoints[:,5,:], keypoints[:,6,:] = copy[:,6,:], copy[:,5,:]
            keypoints[:,7,:], keypoints[:,8,:] = copy[:,8,:], copy[:,7,:]
            keypoints[:,9,:], keypoints[:,10,:] = copy[:,10,:], copy[:,9,:]
            keypoints[:,11,:], keypoints[:,12,:] = copy[:,12,:], copy[:,11,:]
            keypoints[:,13,:], keypoints[:,14,:] = copy[:,14,:], copy[:,13,:]
            keypoints[:,15,:], keypoints[:,16,:] = copy[:,16,:], copy[:,15,:]
        
            return { 'image' : image, 'image_stg_input': ImageOps.mirror(sample['image_stg_input']) ,'keypoints' : keypoints }
        else: return sample

class FlipUD(object):
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']
        
        if np.random.random() > (1-self.p):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            w, h = image.size
            keypoints[:, :, 1][keypoints[:, :, 2]>0] = h - keypoints[:, :, 1][keypoints[:, :, 2]>0]
            copy = keypoints.copy()
            keypoints[:,1,:], keypoints[:,2,:] = copy[:,2,:], copy[:,1,:]
            keypoints[:,3,:], keypoints[:,4,:] = copy[:,4,:], copy[:,3,:]
            keypoints[:,5,:], keypoints[:,6,:] = copy[:,6,:], copy[:,5,:]
            keypoints[:,7,:], keypoints[:,8,:] = copy[:,8,:], copy[:,7,:]
            keypoints[:,9,:], keypoints[:,10,:] = copy[:,10,:], copy[:,9,:]
            keypoints[:,11,:], keypoints[:,12,:] = copy[:,12,:], copy[:,11,:]
            keypoints[:,13,:], keypoints[:,14,:] = copy[:,14,:], copy[:,13,:]
            keypoints[:,15,:], keypoints[:,16,:] = copy[:,16,:], copy[:,15,:]
        
            return { 'image' : image, 'image_stg_input': ImageOps.flip(sample['image_stg_input']) ,'keypoints' : keypoints }
        else: return sample
        
class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.tfm = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, sample):
        image = self.tfm(sample['image'])
        return { 'image' : image, 'image_stg_input': sample['image_stg_input'], 'keypoints': sample['keypoints'] }

class RandomGrayscale(object):
    def __init__(self, p=0.33):
        self.tfm = transforms.RandomGrayscale(p=p)
        self.gs = transforms.Grayscale(num_output_channels=3)
    
    def __call__(self, sample):
        image = self.tfm(sample['image'])
        if(len(image.getbands())<3):
            image = self.gs(image)
            sample['image_stg_input'] = self.gs(sample['image_stg_input'])
        return { 'image' : image, 'image_stg_input' : sample['image_stg_input'], 'keypoints': sample['keypoints'] }

class RandomRotateImgAndKeypoints(object):
    def __init__(self, deg=30, p=0.9):
        self.deg = deg
        self.p = p
    
    def __rotate__(self, origin, keypoints, deg, sz):
        ox, oy = origin
        theta = np.math.radians(-deg) #-deg since we measure y,x from top left and not w/2,h/2
        X = keypoints[:,:,0][keypoints[:,:,2]>0]
        Y = keypoints[:,:,1][keypoints[:,:,2]>0]
        
        keypoints[:,:,0][keypoints[:,:,2]>0] = ox + (np.math.cos(theta)*(X - ox) - np.math.sin(theta)*(Y - oy)) 
        keypoints[:,:,1][keypoints[:,:,2]>0] = oy + (np.math.sin(theta)*(X - ox) + np.math.cos(theta)*(Y - oy)) 
        
        inds = np.logical_or(np.any((keypoints[:,:,:2]<0), axis=2), np.any((keypoints[:,:,:2]>sz), axis=2))
        keypoints[inds,:] = np.array([0,0,0])
        return keypoints
    
    def __call__(self, sample):
        if(np.random.random()>(1-self.p)):
            image = sample['image']
            keypoints = sample['keypoints'].copy()
            rand_deg = np.random.randint(-1*self.deg, self.deg+1)
            image = image.rotate(rand_deg)
            w, h = image.size
            res = self.__rotate__((w/2, h/2), keypoints, rand_deg, h)
            return { 'image' : image, 'image_stg_input' : sample['image_stg_input'].rotate(rand_deg) ,'keypoints' : res }
        else:
            return sample

class ToTensor(object):
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
    
    def __call__(self, sample):
        return { 'image' : self.ToTensor(sample['image']),
                 'image_stg_input' : self.ToTensor(sample['image_stg_input']),
                 'pafs' : torch.tensor(sample['pafs'], dtype=torch.float),
                 'PAF_BINARY_IND' : torch.tensor(sample['PAF_BINARY_IND'], dtype=torch.uint8),
                 'heatmaps' : torch.tensor(sample['heatmaps'], dtype=torch.float),
                 'HM_BINARY_IND' : torch.tensor(sample['HM_BINARY_IND'], dtype=torch.uint8),
                }

class NormalizeImg(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)
    
    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        return sample

class UnNormalizeImgBatch(object):
    def __init__(self, mean, std):
        self.mean = mean.reshape((1,3,1,1))
        self.std = std.reshape((1,3,1,1))
    
    def __call__(self, batch):
        return (batch*self.std) + self.mean

class Resize(object):
    def __init__(self, size=368):
        self.size = size
    
    def __call__(self, im):
        if(im.height > im.width):
            w = int(self.size*im.width/im.height)
            h = self.size
            pad_val = int((self.size-w)/2)
            pad = (self.size-w-pad_val,0,pad_val,0)
        else:
            h = int(self.size*im.height/im.width)
            w = self.size
            pad_val = int((self.size-h)/2)
            pad = (0,self.size-h-pad_val,0,pad_val)
        return ImageOps.expand(im.resize((w,h),resample=Image.BILINEAR), pad)
