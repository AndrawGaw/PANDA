# -*- coding: utf-8 -*-  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
        self.fc_dict = {}  # �洢��ͬͨ������ fc �㣬�����ظ�����

    def forward(self, features):
        B, C, _, _ = features.shape
        
        # �����ǰͨ������ fc �����ڣ��ʹ���һ��
        if C not in self.fc_dict:
            self.fc_dict[C] = nn.Linear(C, C * 3 * 3).to(features.device)

        fc = self.fc_dict[C]  # ѡ���Ӧ�� fc ��
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(B, C)
        conv_kernel = fc(pooled_features)
        conv_kernel = conv_kernel.view(B, C, 3, 3)
        
        return conv_kernel

def dynamic_group_conv(proto, kernel):
    B, C, H, W = proto.shape  # ��ȡ batch, channel, height, width
    
    # ���� kernel ��״Ϊ [B*C, 1, 3, 3]
    kernel = kernel.view(B * C, 1, 3, 3)
    # ���� proto ��״Ϊ [1, B*C, H, W]
    proto = proto.view(1, B * C, H, W)

    # ���з�������ÿ��ͨ������Ӧ��һ�� 3x3 �����
    output = F.conv2d(proto, kernel, padding=1, groups=B * C)
    
    # ������ԭʼ��״
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
        # ��ȡ������豸��CPU �� GPU��
        device = x.device
        print(f"device: {device}")  
        
        # x ����״�� [B, C, M, M]
        B, C, M, N = x.size()  # ��ȡ�������״
        
        # ʹ��1x1������ɾ���ˣ�����ͨ����Ӧ������� C һ��
        conv = nn.Conv2d(C, C * self.kernel_size * self.kernel_size, kernel_size=1)
        
        # ��������Ȩ���Ƶ�������豸��
        conv.to(device)
        
        # �����״: [B, C * 3 * 3, M, M]
        kernel = conv(x)
        
        # ����Ϊ [B, C, 3, 3] �ľ����
        kernel = kernel.view(kernel.size(0), kernel.size(1) // (self.kernel_size * self.kernel_size), self.kernel_size, self.kernel_size)
        return kernel

class EdgeDetectionPromote(nn.Module):
    def __init__(self):
        super(EdgeDetectionPromote, self).__init__()

        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda()  
        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1).cuda()
        
    
        
    def forward(self, x):
        
        # print(f"self.sobel_x.shape: {self.sobel_x.shape}")
        # print(f"self.sobel_y.shape: {self.sobel_y.shape}")
        grad_x = F.conv2d(x, self.sobel_x, padding=1,groups=3)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)

        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Thresholding to create binary edge map
        # edge_map = grad_magnitude > 0.4  # Threshold (adjustable)
        edge_map = grad_magnitude > 0.2
        
        edge_map = edge_map.max(dim=1,keepdim=True)[0]
        # print(f"edge_map shape: {edge_map.shape}")
        
        return edge_map.float()

class EdgePromptFusion(nn.Module):
    def __init__(self):
        super().__init__()
        

        self.edge_encoder = None
        self.attention_gen = None
        self.final_conv = None

    

    def forward(self, q_feat, proto_feat, edge_map):
        query_feat_channels = q_feat.shape[1]
        attention_channels = query_feat_channels//4
        encoded_edge_channels = query_feat_channels//8
        
        if self.edge_encoder is None:
            self.edge_encoder = nn.Sequential(
                nn.Conv2d(1, encoded_edge_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(encoded_edge_channels),
                nn.ReLU(inplace=True)).cuda()
            
            self.attention_gen = nn.Sequential(
                nn.Conv2d(query_feat_channels, attention_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(attention_channels, 1, kernel_size=1),
                nn.Sigmoid()).cuda()
            
            self.final_conv = nn.Conv2d(encoded_edge_channels +query_feat_channels + proto_feat.shape[1], 256, kernel_size=1).cuda()

        edge_encoded = self.edge_encoder(edge_map)  # shape: (B, encoded_edge_channels, H, W)
        # attn_input = torch.cat([q_feat, proto_feat], dim=1)
        edge_weight = self.attention_gen(q_feat)  # shape: (B, 1, H, W)
        edge_weighted = edge_encoded * edge_weight  # shape: (B, encoded_edge_channels, H, W)
        fusion_input = torch.cat([q_feat, proto_feat, edge_weighted], dim=1)
        output = self.final_conv(fusion_input)  # shape: (B, out_channels, H, W)
        
        return output


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
        
        
        # �1�7�1�7�1�7�1�7 layer0 �1�7�1�7 layer4 �1�7�0�4�1�7�1�7�1�7
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


class SegHeadV7(nn.Module):
    def __init__(self, num_classes=2, in_channels=512, bilinear=True):
        super().__init__()
        
        # ��ѧϰ�ĳ߶�Ȩ�أ���ѡ��
        self.w_q4 = nn.Parameter(torch.tensor(1.0))
        self.w_q8 = nn.Parameter(torch.tensor(1.0))
        self.w_q16 = nn.Parameter(torch.tensor(1.0))
        # ===== ��֧һ��FPN =====
        self.fpn_fusion = nn.Conv2d(512+1024+2048, 512 , 3, padding=1)
        self.fpn_cls = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, num_classes, 1)
        )

        
        self.up1 = self.Up(2048 + 1024, 1024, bilinear=bilinear)
        self.up2 = self.Up(1024 + 512, 512, bilinear=bilinear)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1),
            nn.Sigmoid()
        )
        self.reduction = 8
        self.dilation = 2
       # �ռ�ע������֧
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // self.reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // self.reduction, in_channels //self.reduction, kernel_size=3, padding=self.dilation, dilation=self.dilation),
            nn.BatchNorm2d(in_channels // self.reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // self.reduction, 1, kernel_size=1)
        )
     
        self.q4_aug = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )

      

    def forward(self, q16, q8, q4, proto4):
        
        q4_aug = self.q4_aug(q4)
        
         # === ����ԭ�����ƶ���ǿ ===
        if proto4 is not None:
            # ���� q4_aug �� proto_map �� cosine ���ƶ�
            sim = F.cosine_similarity(q4_aug, proto4, dim=1, eps=1e-6).unsqueeze(1)  # [B, 1, H, W]
            q4_aug = q4_aug * (1 + sim)  # ����������Ӧ��ǿ
        
        q16 = q16 * self.w_q16
        q8 = q8 * self.w_q8
        q4 = q4 * self.w_q4
        
        q8_up = self.up1(q16, q8)
        q4_up = self.up2(q8, q4)
 
        fpn_feats = [F.interpolate(f, size=q4.shape[-2:], mode='bilinear', align_corners=False) for f in [q4_up, q8_up, q16]]
        fpn_fused = torch.cat(fpn_feats, dim=1)
        fpn_fused = F.relu(self.fpn_fusion(fpn_fused))
    
    
        attn = self.spatial_attn(q4_aug)
        atten_fused = fpn_fused * attn
        
        out = self.fpn_cls(atten_fused)  # ����Ԥ��

        return out

    class Up(nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                mid_channels = in_channels // 2
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                mid_channels = in_channels // 2

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x1, x2):
            x1 = self.up(x1)
            diff_y = x2.size(2) - x1.size(2)
            diff_x = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)




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
        self.edge_detection = EdgeDetectionPromote()
        self.edge_prompt_4 = EdgePromptFusion()
        self.edge_prompt_8 = EdgePromptFusion()
        self.edge_prompt_16 = EdgePromptFusion()
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
            
        # self.fusion4 = nn.Conv2d(1024+1, 256, 1)
        # self.fusion8 = nn.Conv2d(2048+1, 256, 1)
        # self.fusion16 = nn.Conv2d(4096+1, 256, 1)
        self.fusion4 = nn.Conv2d(1024, 512, 1)
        self.fusion8 = nn.Conv2d(2048, 1024, 1)
        self.fusion16 = nn.Conv2d(4096, 2048, 1)
        self.adoptive_conv_kernel = DynamicConvKernel()
        
        self.cls = SegHeadV7()
        # self.cls = nn.Sequential(
        #         nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=0.1),                 
        #         nn.Conv2d(reduce_dim, classes, kernel_size=1)
        #     ) 
            
                
    def forward(self, supp_img, supp_mask, query_img, query_mask=None):
        # print(f"supp_img.shape=",supp_img.shape)
        B, N, C, H, W = supp_img.shape     
        proto4_list = []
        proto8_list = []
        proto16_list = []
        for i in range(N):
            single_supp_img = supp_img[:, i, :, :, :].squeeze(1)  # �0�2�1�7�1�7�1�7�1�7�0�8�0�2 (B, 3, 200, 200)
            
            single_supp_mask = supp_mask[:, i, :, :].unsqueeze(1)  # �0�2�1�7�1�7�1�7�1�7�0�8�0�2 (B, 1, 200, 200)
            
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
         
        
        # edge_map = self.edge_detection(query_img)
        # edge_map4 = F.interpolate(edge_map, size=q4.shape[-2:], mode='bilinear', align_corners=False)
        # edge_map8 = F.interpolate(edge_map, size=q8.shape[-2:], mode='bilinear', align_corners=False)
        # edge_map16 = F.interpolate(edge_map, size=q16.shape[-2:], mode='bilinear', align_corners=False)
            
        # kernal4 = self.adoptive_conv_kernel(proto4)
        # kernal8 = self.adoptive_conv_kernel(proto8) 
        # kernal16 = self.adoptive_conv_kernel(proto16)   
        
        # # # freq_q4 = self.frequency_domain_conv(q4, kernal4)
        # # # freq_q8 = self.frequency_domain_conv(q8, kernal8)
        # # # freq_q16 = self.frequency_domain_conv(q16, kernal16)
        # d_q4= dynamic_group_conv(q4, kernal4)
        # d_q8= dynamic_group_conv(q8, kernal8)
        # d_q16= dynamic_group_conv(q16, kernal16)
        # # # print(f"freq_q4 shape: {freq_q4.shape}")
        # # # print(f"freq_q8 shape: {freq_q8.shape}")  
        # # # print(f"freq_q16 shape: {freq_q16.shape}")    
       
        # alpha = 0.05
        # # # alpha = 0.5
        # weight_q4 = (1 - alpha) * proto4 + alpha * d_q4 
        # weight_q8 = (1 - alpha) * proto8 + alpha * d_q8
        # weight_q16 = (1 - alpha) * proto16 + alpha * d_q16       
        
        q4 = F.relu(self.fusion4(torch.cat([q4, proto4], dim=1)))
        q8 = F.relu(self.fusion8(torch.cat([q8, proto8], dim=1)))
        q16 = F.relu(self.fusion16(torch.cat([q16, proto16], dim=1)))
        # q4 = F.relu(self.fusion4(torch.cat([q4, proto4], dim=1)))
        # q8 = F.relu(self.fusion8(torch.cat([q8, proto8], dim=1)))
        # q16 = F.relu(self.fusion16(torch.cat([q16, proto16], dim=1)))
        # print(f"q4 shape: {q4.shape}")
        # print(f"q8 shape: {q8.shape}")
        # print(f"q16 shape: {q16.shape}")
        # q4 = F.relu(self.fusion4(torch.cat([q4, weight_q4, edge_map4], dim=1)))
        # q8 = F.relu(self.fusion8(torch.cat([q8, weight_q8, edge_map8], dim=1)))
        # q16 = F.relu(self.fusion16(torch.cat([q16, weight_q16, edge_map16], dim=1)))
        
        # q4 = self.edge_prompt_4(q4, proto4, edge_map4)
        # q8 = self.edge_prompt_8(q8, proto8, edge_map8)
        # q16 = self.edge_prompt_16(q16, proto16, edge_map16)
    
        
      
        # fused_feat = self.multi_scale([q4, q8, q16])
       
        # out = self.cls(fused_feat)
        out = self.cls(q16, q8, q4, proto4)
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