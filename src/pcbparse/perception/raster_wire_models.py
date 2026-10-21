"""Inference architectures from the supplied v6 pin/net integration release.
Training and label-processing routines are excluded.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self,inp,out):
        super().__init__(); self.block=nn.Sequential(nn.Conv2d(inp,out,3,padding=1,bias=False),nn.BatchNorm2d(out),nn.ReLU(inplace=True),nn.Conv2d(out,out,3,padding=1,bias=False),nn.BatchNorm2d(out),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class NetUNet(nn.Module):
    def __init__(self):
        super().__init__(); self.d1=DoubleConv(3,32);self.d2=DoubleConv(32,64);self.d3=DoubleConv(64,128);self.d4=DoubleConv(128,256);self.pool=nn.MaxPool2d(2);self.mid=DoubleConv(256,512);self.u4=nn.ConvTranspose2d(512,256,2,2);self.c4=DoubleConv(512,256);self.u3=nn.ConvTranspose2d(256,128,2,2);self.c3=DoubleConv(256,128);self.u2=nn.ConvTranspose2d(128,64,2,2);self.c2=DoubleConv(128,64);self.u1=nn.ConvTranspose2d(64,32,2,2);self.c1=DoubleConv(64,32);self.out=nn.Conv2d(32,3,1)
    def forward(self,x):
        x1=self.d1(x);x2=self.d2(self.pool(x1));x3=self.d3(self.pool(x2));x4=self.d4(self.pool(x3));x=self.c4(torch.cat([self.u4(self.mid(self.pool(x4))),x4],1));x=self.c3(torch.cat([self.u3(x),x3],1));x=self.c2(torch.cat([self.u2(x),x2],1));x=self.c1(torch.cat([self.u1(x),x1],1));return self.out(x)

def predict(model,image,device,size,stride,ctx=None,*,output_channels=None,batch_size=None,stats=None):
    """Bound tile batches and retain only requested output channels on the host."""
    if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('raster input must be RGB uint8')
    if type(size) is not int or type(stride) is not int or not 0<stride<=size:
        raise ValueError('invalid raster tiling')
    channels=tuple(range(3)) if output_channels is None else tuple(output_channels)
    if not channels or len(set(channels))!=len(channels) or any(type(c) is not int or c not in range(3) for c in channels):
        raise ValueError('invalid raster output channels')
    if batch_size is None:
        # Preserve the validated batch-one checkpoint numerics across GPUs.
        # Wider batches remain explicit experiments: backend kernel choices
        # can move marginal probabilities across the quantization boundary.
        batch_size=1
    if type(batch_size) is not int or not 1<=batch_size<=4:
        raise ValueError('raster batch size must be between one and four')
    h,w=image.shape[:2]
    if not h or not w:raise ValueError('empty raster input')
    total=np.zeros((len(channels),h,w),np.float32);count=np.zeros((h,w),np.float32)
    tops=list(range(0,max(h-size,0)+1,stride));lefts=list(range(0,max(w-size,0)+1,stride))
    if tops[-1]!=max(h-size,0):tops.append(max(h-size,0))
    if lefts[-1]!=max(w-size,0):lefts.append(max(w-size,0))
    tiles=[(top,left) for top in tops for left in lefts]
    if stats is not None:
        stats.update(tile_count=len(tiles),batch_size=batch_size,output_channels=list(channels),
                     host_accumulator_bytes=total.nbytes+count.nbytes)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(tiles),batch_size):
            if ctx is not None:ctx.time_budget.check(stage='raster_wire_inference')
            positions=tiles[start:start+batch_size]
            pads=np.zeros((len(positions),size,size,3),np.uint8)
            for index,(top,left) in enumerate(positions):
                patch=image[top:top+size,left:left+size]
                pads[index,:patch.shape[0],:patch.shape[1]]=patch
            if len(positions)==1:
                x=torch.from_numpy(pads[0].transpose(2,0,1)).float().unsqueeze(0).to(device)/255.
            else:
                x=torch.from_numpy(pads.transpose(0,3,1,2)).float().to(device)/255.
            y=torch.sigmoid(model(x))[:,list(channels)].cpu().numpy()
            for index,(top,left) in enumerate(positions):
                ph,pw=min(size,h-top),min(size,w-left)
                total[:,top:top+ph,left:left+pw]+=y[index,:,:ph,:pw]
                count[top:top+ph,left:left+pw]+=1
    np.maximum(count,1,out=count)
    np.divide(total,count[None],out=total)
    return total

class CrossingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).flatten()

class SpatialCrossingClassifier(CrossingClassifier):
    def __init__(self):
        super().__init__()
        # Preserve the crossing position; an off-center nearby dot must not
        # decide whether the central crossing connects.
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((5,5)),nn.Flatten(),nn.Linear(64*25,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))

class CrossingPredictor:
    """Consumes original RGB pixels and predicted crossing coordinates only."""
    def __init__(self,checkpoint,device='cpu'):
        self.device=torch.device(device);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cls=SpatialCrossingClassifier if saved.get('architecture')=='spatial_v2' else CrossingClassifier
        self.model=cls().to(self.device)
        self.model.load_state_dict(saved['model'],strict=True);self.model.eval()
    @torch.inference_mode()
    def __call__(self,image,crossings):
        if not crossings:return []
        padded=np.pad(image,((22,22),(22,22),(0,0)),constant_values=255)
        patches=[padded[c['y']:c['y']+45,c['x']:c['x']+45].transpose(2,0,1) for c in crossings]
        x=torch.from_numpy(np.stack(patches)).float().to(self.device)/255
        return torch.sigmoid(self.model(1-x)).cpu().tolist()
