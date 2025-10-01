# -*- coding: utf-8 -*-  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from torchvision.models import resnet50
import model.resnet as models
import model.vgg as vgg_models

from functools import reduce
from operator import add  
from .base.swin_transformer import SwinTransformer


def WeightedGap(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

def compute_weighted_proto(supp_feat, supp_mask, target_shape):
  
    supp_mask=supp_mask.float()
    
    m = F.interpolate(
        supp_mask, 
        size=target_shape, 
        mode='area'  ).gt(0.5).float() 
    
    proto = WeightedGap(supp_feat, m)
     
    return proto.expand_as(supp_feat)

class DynamicConvKernel(nn.Module):
    def __init__(self):
        super(DynamicConvKernel, self).__init__()
        self.fc_dict = {}  # 存储不同通道数的 fc 层，避免重复创建

    def forward(self, features):
        B, C, _, _ = features.shape
        
        # 如果当前通道数的 fc 不存在，就创建一个
        if C not in self.fc_dict:
            self.fc_dict[C] = nn.Linear(C, C * 3 * 3).to(features.device)

        fc = self.fc_dict[C]  # 选择对应的 fc 层
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(B, C)
        conv_kernel = fc(pooled_features)
        conv_kernel = conv_kernel.view(B, C, 3, 3)
        
        return conv_kernel

def dynamic_group_conv(proto, kernel):
    B, C, H, W = proto.shape  # 读取 batch, channel, height, width
    
    # 调整 kernel 形状为 [B*C, 1, 3, 3]
    kernel = kernel.view(B * C, 1, 3, 3)
    # 调整 proto 形状为 [1, B*C, H, W]
    proto = proto.view(1, B * C, H, W)

    # 进行分组卷积，每个通道独立应用一个 3x3 卷积核
    output = F.conv2d(proto, kernel, padding=1, groups=B * C)
    
    # 调整回原始形状
    output = output.view(B, C, H, W)

    return output



class ConvKernelGenerator(nn.Module):
    def __init__(self, kernel_size=3):
        super(ConvKernelGenerator, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # 获取输入的设备（CPU 或 GPU）
        device = x.device
        print(f"device: {device}")  
        
        # x 的形状是 [B, C, M, M]
        B, C, M, N = x.size()  # 获取输入的形状
        
        # 使用1x1卷积生成卷积核，输入通道数应与输入的 C 一致
        conv = nn.Conv2d(C, C * self.kernel_size * self.kernel_size, kernel_size=1)
        
        # 将卷积层的权重移到输入的设备上
        conv.to(device)
        
        # 输出形状: [B, C * 3 * 3, M, M]
        kernel = conv(x)
        
        # 重塑为 [B, C, 3, 3] 的卷积核
        kernel = kernel.view(kernel.size(0), kernel.size(1) // (self.kernel_size * self.kernel_size), self.kernel_size, self.kernel_size)
        return kernel

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, backbone = "resnet",layers=50, pretrained=True, pretrain_path=None):
        super().__init__()
        from torch.nn import BatchNorm2d as BatchNorm  
        self.backbone = backbone 
        
        if self.backbone == "resnet":
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)

            self.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu1, 
                resnet.conv2, resnet.bn2, resnet.relu2, 
                resnet.conv3, resnet.bn3, resnet.relu3, 
                resnet.maxpool
            )
            self.layer1, self.layer2, self.layer3, self.layer4 = (
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            )
            
                # ?1?7?1?7?1?7?1?7 layer0 ?1?7?1?7 layer4 ?1?7?0?4?1?7?1?7?1?7
            for param in self.layer0.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False
            
            
        elif self.backbone == "vgg":
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        
            # ?1?7?1?7?1?7?1?7 layer0 ?1?7?1?7 layer4 ?1?7?0?4?1?7?1?7?1?7
            for param in self.layer0.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False
        
        elif self.backbone == "swin":
            print('INFO: Using Swin Transformer')
            self.pretrained_path = pretrain_path 
            self.backbone_net = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.backbone_net.load_state_dict(torch.load(self.pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]

            self.extract_feats = self.extract_feats_swinB

            # self.nlayers = [2, 2, 18, 2]
            nbottlenecks = [2, 2, 18, 2]
            
            self.layer4 = nn.Sequential(
                            self.backbone_net.layers[3].blocks[-2],
                            self.backbone_net.layers[3].blocks[-1]
                        )
            self.downsample = self.backbone_net.layers[2].downsample  # PatchMerging 层

            self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
            self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
            self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
     
                    # 冻结所有层
            for param in self.backbone_net.parameters():
                param.requires_grad = False

    def forward(self, img):
        if self.backbone == "resnet" or self.backbone == "vgg":
            x = self.layer0(img)
            fs2 = self.layer1(x) 
            fs4 = self.layer2(fs2)
            fs8 = self.layer3(fs4)
            fs16 = self.layer4(fs8) 

        elif self.backbone == "swin":
            query_feats = self.extract_feats(img)
            fs2 = query_feats[1]
            fs4 = query_feats[3]
            fs8 = query_feats[21]
            fs16 = query_feats[23]  # 12x12

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        return fs2, fs4, fs8 , fs16    
    
    def extract_feats_swinB(self, img):
        r""" Extract input image features """
        feats = []

        _ = self.backbone_net.forward_features(img)
        for feat in self.backbone_net.feat_maps:
            bsz, hw, c = feat.size()
            h = int(hw ** 0.5)
            feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
            feats.append(feat)

        return feats
 


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class FPN(nn.Module):
    def __init__(self, in_channels, output_size=(200, 200)):
        super().__init__()
        self.output_size = output_size
        
        self.fusion_conv = nn.Conv2d(in_channels * 4, in_channels, 3, padding=1)
        
    def forward(self, features):  # features = [P3, P4, P5, P6]
       

        h,w = features[0].shape[-2], features[0].shape[-1]
        for i in range(len(features)):
                features[i] = F.interpolate(features[i], size=(h, w), mode='bilinear', align_corners=False)

        
        fused = torch.cat(features, dim=1)  # [B, 4*in_channels, H, W]
        fused = self.fusion_conv(fused)       # [B, in_channels, H, W] 
        fused = F.relu(fused)      
        
        return fused

    

class PANDA(nn.Module):
    def __init__(self, multi_mode="fpn", output_size=(200, 200), classes=2,\
        reduce_dim=256,criterion=nn.CrossEntropyLoss(ignore_index=255),BatchNorm=nn.BatchNorm2d,backbone="resnet",pretain_path=None):
        super().__init__()
        
        self.multi_mode = multi_mode
       
        self.classes = classes
        self.reduce_dim = reduce_dim 
        self.criterion = criterion  
        self.backbone = backbone
        if self.backbone == "resnet" or self.backbone == "vgg":
            self.output_size = (200, 200)  # ResNet and VGG use 200x200 output size
        elif self.backbone == "swin":
            self.output_size = (384, 384)
        self.pretain_path = pretain_path 
        
        self.feature_extractor = MultiScaleFeatureExtractor(backbone = self.backbone, layers=50, pretrained=True, pretrain_path=self.pretain_path)
        self.frequency_domain_conv = FrequencyDomainConv()
        self.learnable_fusion = True
  
        if self.backbone == "resnet":
            self.fea2_dim = 256
            self.fea4_dim = 512
            self.fea8_dim = 1024
        elif self.backbone == "vgg" :
            self.fea2_dim = 128
            self.fea4_dim = 256
            self.fea8_dim = 512
        elif self.backbone == "swin":
            self.fea2_dim = 128
            self.fea4_dim = 256
            self.fea8_dim = 512

        reduce_dim = 256
    
        
        
        if multi_mode == "fpn":
            self.multi_scale = FPN(in_channels=256)
        else:
            raise ValueError("Invalid multi_mo
        self.fusion2 = nn.Conv2d(2*self.fea2_dim, reduce_dim, 1)
        self.fusion4 = nn.Conv2d(2*self.fea4_dim, reduce_dim, 1)
        self.fusion8 = nn.Conv2d(2*self.fea8_dim, reduce_dim, 1)

        

        self.adoptive_conv_kernel = DynamicConvKernel()
        
        self.alpha_conv= nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU())
        self.merge_conv= nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False), #去掉了先验掩码
            nn.ReLU(inplace=True))                  
        self.beta_conv= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))   
        
        self.cls = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ) 
            
                
    def forward(self, supp_img, supp_mask, query_img, query_mask=None):
       
        B, N, C, H, W = supp_img.shape  
        proto2_list = []   
        proto4_list = []
        proto8_list = []

        for i in range(N):
            single_supp_img = supp_img[:, i, :, :, :].squeeze(1)  # ?0?2?1?7?1?7?1?7?1?7?0?8?0?2 (B, 3, 200, 200)
            
            single_supp_mask = supp_mask[:, i, :, :].unsqueeze(1)  # ?0?2?1?7?1?7?1?7?1?7?0?8?0?2 (B, 1, 200, 200)

            with torch.no_grad():
                s2,s4, s8 ,s16 = self.feature_extractor(single_supp_img)
            proto2 = compute_weighted_proto(s2, single_supp_mask, s2.shape[-2:])
            proto4 = compute_weighted_proto(s4, single_supp_mask, s4.shape[-2:])
            proto8 = compute_weighted_proto(s8, single_supp_mask, s8.shape[-2:])
            proto2_list.append(proto2)   
            proto4_list.append(proto4)
            proto8_list.append(proto8)
   
            

        with torch.no_grad():
            q2,q4, q8 , q16 = self.feature_extractor(query_img)
        
        proto2 = torch.stack(proto2_list, dim=0).mean(dim=0)  # (B, 256, H, W)
        proto4 = torch.stack(proto4_list, dim=0).mean(dim=0)    
        proto8 = torch.stack(proto8_list, dim=0).mean(dim=0) 

        kernal2 = self.adoptive_conv_kernel(proto2)   
        kernal4 = self.adoptive_conv_kernel(proto4)
        kernal8 = self.adoptive_conv_kernel(proto8) 

        d_q2 = dynamic_group_conv(q2, kernal2)
        d_q4= dynamic_group_conv(q4, kernal4)
        d_q8= dynamic_group_conv(q8, kernal8)


        alpha = 0.05
        weight_q2 = (1 - alpha) * proto2 + alpha * d_q2
        weight_q4 = (1 - alpha) * proto4 + alpha * d_q4 
        weight_q8 = (1 - alpha) * proto8 + alpha * d_q8
    
        
        

        q2 = F.relu(self.fusion2(torch.cat([weight_q2, q2], dim=1)))
        q4 = F.relu(self.fusion4(torch.cat([weight_q4, q4], dim=1)))
        q8 = F.relu(self.fusion8(torch.cat([weight_q8, q8], dim=1)))

 
        pyramid_feat_list = []
        # out_list = []

        for idx, feat in enumerate([q2, q4, q8]):
            if idx != 0:
                 # 第一层直接处理
           
                pre_feat = pyramid_feat_list[idx - 1]
                pre_feat = F.interpolate(pre_feat, size=feat.shape[-2:], mode='bilinear', align_corners=True)
                rec_feat = torch.cat([feat, pre_feat], dim=1)
                feat = self.merge_conv(rec_feat) + feat

            merge_feat = self.beta_conv(feat) + feat
            pyramid_feat_list.append(merge_feat)
        
        fused_feat = self.multi_scale(pyramid_feat_list) 

       
        out = self.cls(fused_feat)
        
        out= F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)
        

     
        if self.training:
            assert query_mask is not None, "query_mask can't be None in training mode"
            main_loss = self.criterion(out, query_mask.long())  
            return out.max(1)[1], main_loss
        else:
            return out  

    def predict_mask_nshot(self, query_img, support_imgs, support_masks, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(support_imgs,support_masks,query_img)
            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask