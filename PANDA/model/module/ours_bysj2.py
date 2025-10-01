# -*- coding: utf-8 -*-  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from torchvision.models import resnet50
import model.resnet as models

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

class FrequencyDomainConv(nn.Module):
    def __init__(self, learnable_fusion=True):
        super(FrequencyDomainConv, self).__init__()
        self.learnable_fusion = learnable_fusion
        if self.learnable_fusion:
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, feature_map, kernel):
        B, C, H, W = feature_map.shape
        _, _, kH, kW = kernel.shape

        kernel_padded = F.pad(kernel, [0, W - kW, 0, H - kH])
        feat_fft = torch.fft.rfft2(feature_map, dim=(-2, -1))
        kernel_fft = torch.fft.rfft2(kernel_padded, dim=(-2, -1))
        out_fft = feat_fft * kernel_fft
        out = torch.fft.irfft2(out_fft, s=(H, W))

        if self.learnable_fusion:
            return (1 - self.alpha) * feature_map + self.alpha * out
        else:
            return out


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
    def __init__(self, layers=50, pretrained=True):
        super().__init__()
        
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
        
        
        # ���� layer0 �� layer4 �Ĳ���
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
        


    def forward(self, img):
        x = self.layer0(img)
        x = self.layer1(x) 
        fs4 = self.layer2(x)
        # print(f"layer2 output shape: {fs4.shape}")  
        fs8 = self.layer3(fs4)
        # supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        fs16 = self.layer4(fs8) 
        
        return fs4, fs8, fs16       
       

class FPN(nn.Module):
    def __init__(self, in_channels, output_size=(200, 200)):
        super().__init__()
        self.output_size = output_size
        
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, 3, padding=1)
        
    def forward(self, features):  # features = [P3, P4, P5, P6]
       
        # h, w = features[0].shape[-2], features[0].shape[-1]
        h,w = features[0].shape[-2], features[0].shape[-1]
        for i in range(len(features)):
                features[i] = F.interpolate(features[i], size=(h, w), mode='bilinear', align_corners=False)

        
        fused = torch.cat(features, dim=1)  # [B, 4*in_channels, H, W]
        fused = self.fusion_conv(fused)       # [B, in_channels, H, W] 
        fused = F.relu(fused)      
        
        return fused

class ASPP(nn.Module):
    def __init__(self, in_channels, output_size=(224, 224)):  # ԭͼ�ߴ�
        super().__init__()
        self.output_size = output_size
        
        # �������;������������ݶ�߶Ȳ㼶��̬���������ʣ�
        self.dilation_rates = [4, 2, 1]  # ��Ӧq4, q8, q16�������ʣ���������Ϊ224��
        
        # 1��1�����ά��
        self.conv1x1 = nn.Conv2d(256 * 3, 256, 1)  # �ϲ�3���㼶������ά
        
        # ���;���㣨���ָ��ߴ磩
        self.dilated_convs = nn.ModuleList()
        for rate in self.dilation_rates:
            conv = nn.Conv2d(256, 256, kernel_size=3, padding=rate, dilation=rate)
            self.dilated_convs.append(conv)
        
        # ȫ��ƽ���ػ�����ѡ��
        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)
        
    def forward(self, features):
        # features = [q4, q8, q16]����״�ֱ�Ϊ [B,256,56,56], [B,256,28,28], [B,256,14,14]
        
        # ��һ����ͨ�����;���ָ����㼶������ԭͼ�ߴ�
        recovered_feats = []
        for i, feat in enumerate(features):
            h, w = feat.shape[-2], feat.shape[-1]
            target_h, target_w = self.output_size
            # �������������ʣ����ݵ�ǰ�㼶��ԭͼ�ߴ綯̬������
            rate = self.dilation_rates[i]
            # Ӧ�����;���ָ��ߴ�
            recovered = self.dilated_convs[i](feat)
            recovered = F.interpolate(recovered, (target_h, target_w), mode='bilinear', align_corners=False)
            recovered_feats.append(recovered)
        
        # �ڶ�����ͨ��ƴ����1��1��ά
        fused = torch.cat(recovered_feats, dim=1)  # [B, 768, 224, 224]
        fused = self.conv1x1(fused)  # [B, 256, 224, 224]
        fused = F.relu(fused)
        
       
        x = self.global_avg(fused)  # [B, 256, 1, 1]
        x = self.fc(x)             # [B, 256]
        x = F.relu(x)
        
        return x.unsqueeze(1)  # [B, 1, 1, 1]

class TransformerModule(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.query = nn.Conv2d(in_channels, 256, 1)
        self.key = nn.Conv2d(in_channels, 256, 1)
        self.value = nn.Conv2d(in_channels, 256, 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, F4, F8, F16):
        b, c, h, w = F4.shape
        F16_up = F.interpolate(F16, size=(h, w), mode='bilinear', align_corners=True)
        
        q = self.query(F4).view(b, 256, -1).permute(0, 2, 1)
        k = self.key(F16_up).view(b, 256, -1)
        v = self.value(F16_up).view(b, 256, -1).permute(0, 2, 1)
        
        attn=torch.bmm(q,k)/(256**0.5)
        attn = self.softmax(attn)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(b, 256, h, w)
        return F4 + out
    

class MultiScaleSegmentation(nn.Module):
    def __init__(self, multi_mode="fpn", output_size=(200, 200), classes=2,\
        reduce_dim=256,criterion=nn.CrossEntropyLoss(ignore_index=255),BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.feature_extractor = MultiScaleFeatureExtractor()
        self.multi_mode = multi_mode
        self.output_size = output_size
        self.classes = classes
        self.reduce_dim = reduce_dim 
        self.criterion = criterion   
        self.frequency_domain_conv = FrequencyDomainConv()
        self.learnable_fusion = True
        from torch.nn import BatchNorm2d as BatchNorm
        models.resnet50.BatchNorm = BatchNorm
        
        
    
        self.compress4 = nn.Conv2d(256, 256, 1)
        self.compress8 = nn.Conv2d(512, 256, 1)
        self.compress16 = nn.Conv2d(1024, 256, 1)
        
        
        if multi_mode == "fpn":
            self.multi_scale = FPN(in_channels=256)
        else:
            raise ValueError("Invalid multi_mode")
            
        self.fusion4 = nn.Conv2d(1024, 256, 1)
        self.fusion8 = nn.Conv2d(2048, 256, 1)
        self.fusion16 = nn.Conv2d(4096, 256, 1)
        
        # self.dynamic_conv = DynamicConv(in_channels=256, out_channels=256, kernel_size=3)
        self.adoptive_conv_kernel = DynamicConvKernel()
        
    
        self.cls = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ) 
            
                
    def forward(self, supp_img, supp_mask, query_img, query_mask=None):
        # print(f"supp_img.shape=",supp_img.shape)
        B, N, C, H, W = supp_img.shape     
        proto4_list = []
        proto8_list = []
        proto16_list = []
        for i in range(N):
            single_supp_img = supp_img[:, i, :, :, :].squeeze(1)  # ȷ����״Ϊ (B, 3, 200, 200)
            
            single_supp_mask = supp_mask[:, i, :, :].unsqueeze(1)  # ȷ����״Ϊ (B, 1, 200, 200)
            # print(f"single_supp_img shape: {single_supp_img.shape}")
            # print(f"single_supp_mask shape: {single_supp_mask.shape}")
            with torch.no_grad():
                s4, s8, s16 = self.feature_extractor(single_supp_img)
            proto4 = compute_weighted_proto(s4, single_supp_mask, s4.shape[-2:])
            proto8 = compute_weighted_proto(s8, single_supp_mask, s8.shape[-2:])
            proto16 = compute_weighted_proto(s16, single_supp_mask, s16.shape[-2:])    
            proto4_list.append(proto4)
            proto8_list.append(proto8)
            proto16_list.append(proto16)        
            

        with torch.no_grad():
            q4, q8, q16 = self.feature_extractor(query_img)
        
        if N > 1:
            proto4 = proto4_list[0]
            proto8 = proto8_list[0] 
            proto16 = proto16_list[0]
            proto4 = torch.stack(proto4_list, dim=0).mean(dim=0)    
            proto8 = torch.stack(proto8_list, dim=0).mean(dim=0)
            proto16 = torch.stack(proto16_list, dim=0).mean(dim=0)
               
       
        # print(f"proto4 shape: {proto4.shape}")
        # print(f"proto8 shape: {proto8.shape}")  
        # print(f"proto16 shape: {proto16.shape}")     
        kernal4 = self.adoptive_conv_kernel(proto4)
        kernal8 = self.adoptive_conv_kernel(proto8) 
        kernal16 = self.adoptive_conv_kernel(proto16)   
        
        # freq_q4 = self.frequency_domain_conv(q4, kernal4)
        # freq_q8 = self.frequency_domain_conv(q8, kernal8)
        # freq_q16 = self.frequency_domain_conv(q16, kernal16)
        d_q4= dynamic_group_conv(q4, kernal4)
        d_q8= dynamic_group_conv(q8, kernal8)
        d_q16= dynamic_group_conv(q16, kernal16)
        # print(f"freq_q4 shape: {freq_q4.shape}")
        # print(f"freq_q8 shape: {freq_q8.shape}")  
        # print(f"freq_q16 shape: {freq_q16.shape}")    
       
        alpha = 0.05
        # alpha = 0.5
        weight_q4 = (1 - alpha) * proto4 + alpha * d_q4 
        weight_q8 = (1 - alpha) * proto8 + alpha * d_q8
        weight_q16 = (1 - alpha) * proto16 + alpha * d_q16       
        
        
        # q4 = F.relu(self.fusion4(torch.cat([d_q4, proto4], dim=1)))
        # q8 = F.relu(self.fusion8(torch.cat([d_q8, proto8], dim=1)))
        # q16 = F.relu(self.fusion16(torch.cat([d_q16, proto16], dim=1)))
        q4 = F.relu(self.fusion4(torch.cat([weight_q4, q4], dim=1)))
        q8 = F.relu(self.fusion8(torch.cat([weight_q8, q8], dim=1)))
        q16 = F.relu(self.fusion16(torch.cat([weight_q16, q16], dim=1)))
        
      
        fused_feat = self.multi_scale([q4, q8, q16])
       
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