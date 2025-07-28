import torch
import torch.nn as nn
import numpy as np

import utils.utils
import model.transformer_encoder as Transformer
import model.losses as losses
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
    
class DetectionCircle(nn.Module):
    def __init__(self, anchors, image_size: int, obj: str, grid_size: int, device: str):
        super(DetectionCircle, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.image_size = image_size
        
        self.diou = losses.BboxLoss(obj)
        self.ldiou = 1 #lambda diou
        self.fl = losses.FocalLoss()
        self.lfl = 1 #lambda diou
        
        self.device = device
        self.grid_size = grid_size
        self.metrics = {}
        
        # Calculate offsets for each grid
        self.stride = self.image_size / self.grid_size #patch size
        self.cx = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        self.cy = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        
        self.scaled_anchors = torch.as_tensor([(a_r / self.stride) for a_r in self.anchors],
                                         dtype=torch.float, device=self.device)# scale relative to patch size
        self.pr = self.scaled_anchors.view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets, gamma=2.0, alpha=0.25):
        """
            Compute the prediction and loss of the model output
             x shape : batch_size, anchors*(box_coordinates+score+n_class), grid_size, grid_size
             target shape: batch_size, box_coordinates+score+n_class, grid_size, grid_size
        """
        batch_size = x.size(0)
        #grid_size = x.size(2)

        prediction = (
            x.view(batch_size, self.num_anchors, 4, self.grid_size, self.grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )
        # batch_size, n anchors, grid size, grid size, box

        # Get outputs
        sigmoid_tx = torch.sigmoid(prediction[..., 0])  # Center x
        sigmoid_ty = torch.sigmoid(prediction[..., 1])  # Center y
        tr = prediction[..., 2]  # Width
        pred_conf = torch.sigmoid(prediction[..., 3])  # Object confidence (objectness)
        #pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :3], device=self.device)
        pred_boxes[..., 0] = sigmoid_tx + self.cx
        pred_boxes[..., 1] = sigmoid_ty + self.cy
        pred_boxes[..., 2] = self.pr * torch.exp(tr)
        
        pred_boxes = pred_boxes/self.grid_size #prediction in range of 0 to 1
        pred = (pred_boxes.view(batch_size, -1, 3) * self.image_size, #prediction in range of 0 to img_size
                pred_conf.view(batch_size, -1, 1))
        output = torch.cat(pred, -1)
        
        if targets is None:
            #output shape : batch_size, n boxes detections (grid_size**2), 4 (xc, yc, r, conf)
            return output, 0
        #aligned_targets, iou_scores, obj_mask, no_obj_mask = utils.utils.build_targets_circle(
        aligned_targets, pred_boxes, obj_mask, no_obj_mask = utils.utils.build_targets_circle(
            pred_boxes=pred_boxes, #in range of 0 to 1
            target=targets, #range of 0-1
            anchors=self.scaled_anchors,
            device=self.device,
            iou_type=True # if true the aligned targets are the same as input but aligned
        )
        
        #target ordered for comparison with corresponding pred_boxes
        loss_bbox = self.diou(pred_boxes, aligned_targets, obj_mask)
        #loss_bbox, loss_empty, loss_missed = self.diou(pred_boxes, aligned_targets, obj_mask)
        
        #loss_l1 = F.l1_loss(pred_boxes, aligned_targets)
       
        loss_conf = self.fl(pred=prediction[..., 3], label=obj_mask.float(), gamma=gamma, alpha=alpha)
        
        loss_layer = self.ldiou*loss_bbox + self.lfl*loss_conf #+ loss_empty + loss_missed#+ loss_cls

        # Write loss and metrics
        self.metrics = {
            "loss_bbox": loss_bbox.detach(),
            "loss_conf": loss_conf.detach(),
            "loss_layer": loss_layer.detach(),
        }
        
        return output, loss_layer
    
class DetectionRectangle(nn.Module):
    def __init__(self, anchors, image_size: int, obj: str, grid_size: int, device: str):
        super(DetectionRectangle, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.image_size = image_size
        
        self.diou = losses.BboxLoss(obj)
        self.ldiou = 1 #lambda diou
        self.fl = losses.FocalLoss()
        self.lfl = 1 #lambda diou

        self.device = device
        self.grid_size = grid_size
        self.metrics = {}
        
        # Calculate offsets for each grid
        self.stride = self.image_size / self.grid_size #patch size
        self.cx = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        self.cy = torch.arange(grid_size, dtype=torch.float, device=self.device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        self.scaled_anchors = torch.as_tensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=self.device)# scale relative to patch s
        self.pw = self.scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        self.ph = self.scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))
        
        print(self.scaled_anchors)
        #self.box = []

    def forward(self, x, targets, gamma=1.5, alpha=0.75):
        #Compute the prediction and loss of the model output
        # x shape : batch_size, anchors*(box_coordinates+score+n_class), grid_size, grid_size
        # target shape: batch_size, box_coordinates+score+n_class, grid_size, grid_size

        batch_size = x.size(0)

        prediction = (
            x.view(batch_size, self.num_anchors, 5, self.grid_size, self.grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )
        # batch_size, n anchors, grid size, grid size, classes + box
        
        # Get outputs
        sigmoid_tx = torch.sigmoid(prediction[..., 0])  # Center x
        sigmoid_ty = torch.sigmoid(prediction[..., 1])  # Center y
        tw = prediction[..., 2]  # Width
        th = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        #pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction
        
        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :4], device=self.device)
        pred_boxes[..., 0] = sigmoid_tx + self.cx
        pred_boxes[..., 1] = sigmoid_ty + self.cy
        pred_boxes[..., 2] = self.pw * torch.exp(tw)
        pred_boxes[..., 3] = self.ph * torch.exp(th)
        
        pred_boxes = pred_boxes/self.grid_size #prediction in range of 0 to 1
        pred = (pred_boxes.view(batch_size, -1, 4) * self.image_size, #prediction in range of 0 to img_size
                pred_conf.view(batch_size, -1, 1))
        output = torch.cat(pred, -1)

        if targets is None:
            #output shape : batch_size, n boxes detections (grid_size**2), 4 (xc, yc, w, h, conf)
            #self.box=output[:,10,0:4]
            #output = output[:,10,-1].view(-1, 1)
            return output, 0
        #aligned_targets, iou_scores, obj_mask, no_obj_mask = utils.utils.build_targets_rec(
        aligned_targets, pred_boxes, obj_mask, _ = utils.utils.build_targets_rec(
            pred_boxes=pred_boxes, #in range of 0 to 1
            target=targets, #range of 0-1
            anchors=self.scaled_anchors,
            device=self.device,
            iou_type=True
        )
        #target ordered for comparison with corresponding pred_boxes
        loss_bbox = self.diou(pred_boxes, aligned_targets, obj_mask)
        loss_conf = self.fl(pred=prediction[..., 4], label=obj_mask.float(), gamma=gamma, alpha=alpha)

        loss_layer = self.ldiou*loss_bbox + self.lfl*loss_conf #+ loss_cls

        # Write loss and metrics
        self.metrics = {
            "loss_bbox": loss_bbox.detach(), #.cpu().item() reduce time not to get item 
            "loss_conf": loss_conf.detach(),
            "loss_layer": loss_layer.detach()
        }

        return output, loss_layer
    
class TransformerObjectDetection(nn.Module):
    def __init__(self, image_size, N_channels=3, n_model=512, num_blks=1, anchors=None, obj='cbbox', device='cpu', bitNet=False, gamma=2.0, alpha=0.75, segment=False, token=None):
        super(TransformerObjectDetection, self).__init__()
        #anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
        #           'scale2': [(30, 61), (62, 45), (59, 119)],
        #           'scale3': [(116, 90), (58, 53), (373, 326)]}
        #anchors = {'scale1': [(12, 13), (15, 16), (17, 19)],
        #           'scale2': [(20, 23), (21, 19), (24, 23)],
        #           'scale3': [(24, 27), (28, 29), (33, 34)]}
        #anchors = {'scale2': [(13, 14), (16, 18), (19, 19)],
        #           'scale3': [(21, 23), (25, 26), (31, 32)]}
        assert obj in ['cbbox', 'bbox']
        
        if anchors==None:
            if obj=='bbox':
                anchors = {'scale1': [(21, 21)], 'scale2': [(70, 69)]}
                #anchors = {'scale1': [(70, 69)], 'scale2': [(21, 21)]}
            else:#cbbox
                anchors = {'scale1': [(11)], 'scale2': [(35)]}
                
        #each anchor have 4 values for bboxes, 1 for object confidence, n num classes scores
        if obj == 'bbox':
            final_out_channel = len(anchors[list(anchors.keys())[0]]) * (4 + 1 ) 
        else: #cbbox
            final_out_channel = len(anchors[list(anchors.keys())[0]]) * (3 + 1 ) 
        
        img_size=(image_size, image_size)

        patch_enc1 = 8
        patch_enc2 = 2
        
        img_size2=(image_size//patch_enc1, image_size//patch_enc1)
        
        self.segment=segment
        self.encoder1 = Transformer.Encoder(img_size, N_channels, n_model//2, mult=2, 
                                           patch_size=patch_enc1, num_blks=num_blks, device=device, 
                                           bitNet=bitNet,features=segment)
        grid_size1 = img_size[0]//patch_enc1#only square images
        self.encoder2 = Transformer.Encoder(img_size2, n_model//2, n_model, mult=2, 
                                           patch_size=patch_enc2, num_blks=num_blks, device=device, bitNet=bitNet)
        
        grid_size2 = grid_size1//patch_enc2
        self.conv_final1 = self.make_conv_final(n_model, final_out_channel) #input is the output of enc2
        #self.conv_final1_box = self.make_conv_final_box(n_model, 4) #input is the output of enc2
        #self.conv_final1_cls = self.make_conv_final_cls(n_model, 1) #input is the output of enc2
        
        if obj=='bbox':
            self.detection1 = DetectionRectangle(anchors['scale1'], image_size, obj, grid_size2, device)
        elif obj=='cbbox':
            self.detection1 = DetectionCircle(anchors['scale1'], image_size, obj, grid_size2, device) # diou + fl
            
        #self.conv_residual = self.make_conv(n_model//2, n_model//4, kernel_size=1, stride=1, padding=0)
        self.upsample = self.make_upsample(n_model, n_model//2, scale_factor=patch_enc2)
        
        self.conv_final2 = self.make_conv_final(n_model, final_out_channel)
        #self.conv_final2_box = self.make_conv_final_box(n_model, 4) #input is the output of enc2
        #self.conv_final2_cls = self.make_conv_final_cls(n_model, 1) #input is the output of enc2

        if obj=='bbox':
            self.detection2 = DetectionRectangle(anchors['scale2'], image_size, obj, grid_size1, device)
        elif obj=='cbbox':
            self.detection2 = DetectionCircle(anchors['scale2'], image_size, obj, grid_size1, device)

        #parameters for Focal Loss
        self.gamma = gamma
        self.alpha = alpha

        self.layers = [self.detection1, self.detection2]
        
        #for explainability
        self.box=[]
        self.token=token
        
        self.segment = segment
        if segment:
            self.upsample2 = self.make_upsample2(n_model, n_model//2, scale_factor=patch_enc2)
            self.upsample3 = self.make_upsample2(n_model//2+n_model//4, n_model//4, scale_factor=patch_enc2)
            self.upsample4 = self.make_upsample2(n_model//4+n_model//8, n_model//8, scale_factor=patch_enc2)
            self.conv_final3 = self.make_conv2(n_model//8, 2, kernel_size=3)
        
        self.loss_m = losses.DiceLoss(2)#nn.MSELoss()
        #self.loss_m = losses.HuberLoss()
        #self.loss_m = MultiScaleVegetationLoss()
        
    def forward(self, x, targets=None, y_true=None):
        """
            x shape: batch_size, N channels, img_size, img_size
            targets shape: batch_size, N objects in the batch, coordinates box
        """
        loss = 0
        
        if self.segment:
            x, feats = self.encoder1(x) ##1/8 (28x28)
            feat_1_2 = feats[0]
            feat_1_4 = feats[1]
            #feat_1_8 = feats[2]
        else:
            x = self.encoder1(x) ##1/8 (28x28)
        residual_output = x
        
        x = self.encoder2(x)##1/16 (14x14)
        
        scale1 = self.conv_final1(x)
        
        output1, layer_loss = self.detection1(scale1, targets, self.gamma, self.alpha)
        loss += layer_loss
        
        x = self.upsample(x)
        x = torch.cat((x, residual_output), dim=1)
        
        scale2 = self.conv_final2(x)
        #scale2_box = self.conv_final2_box(x)
        #scale2_cls = self.conv_final2_cls(x)
        #scale2 = torch.cat((scale2_box, scale2_cls), dim=1)
        
        output2, layer_loss = self.detection2(scale2, targets, self.gamma, self.alpha)
        loss += layer_loss
        
        if self.segment:
            x = self.upsample2(x) #1//4
            #print("x upsample2", x.shape)
            x = torch.cat((x, feat_1_4), dim=1)
            #print("x 1/4", x.shape)#x 1/4 torch.Size([64, 96, 56, 56])
            x = self.upsample3(x) #1//2
            #print("x upsample3", x.shape)
            x = torch.cat((x, feat_1_2), dim=1)
            #print("x 1/2", x.shape)
            x = self.upsample4(x) #1
            x = self.conv_final3(x)

        if y_true!=None:
            loss_mask = self.loss_m(x, y_true[:,0,:,:])#dice loss
            #loss_mask = self.loss_m(x, y_true)
            loss += loss_mask

        if self.token!=None:
            #outputs = output1#torch.rand(1, 984, 6)
            outputs = torch.cat([output1, output2], 1)   
            self.box = outputs[:,self.token,0:4]
            outputs = outputs[:, self.token,-1].view(-1, 1)        # for explainability, select a box
        else:
            outputs = torch.cat([output1.detach(), output2.detach()], 1)
        
        if self.segment:
             return (outputs, x) if targets is None else (loss, outputs, x)
        else:
            return outputs if targets is None else (loss, outputs)
    
    def make_conv2(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

        modules = nn.Sequential(module1, module2)#nn.LeakyReLU(negative_slope=0.1))
        return modules
    
    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

        modules = nn.Sequential(module1, module2, nn.GELU())#nn.LeakyReLU(negative_slope=0.1))
        return modules

    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            #self.make_conv(in_channels, in_channels//4, kernel_size=3, padding=1),
            #self.make_conv(in_channels//4, in_channels//8, kernel_size=3, padding=1),
            self.make_conv(in_channels, in_channels//4, kernel_size=3),
            self.make_conv(in_channels//4, in_channels//8, kernel_size=3),
            
            #nn.Conv2d(in_channels//8, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return modules

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules
    def make_upsample2(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=3),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules
    
class TinyViT_det(nn.Module):
    def __init__(self, image_size=224, n_model=512, device='cpu', gamma=2.0, alpha=0.75):
        super().__init__()
        from TinyViT.models.tiny_vit import tiny_vit_5m_224
        self.backbone = tiny_vit_5m_224(pretrained=True).to(device)
        self.backbone.eval()  # Freeze running stats
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        patch_enc1 = 8
        patch_enc2 = 2
        img_size=(image_size, image_size)
        
        grid_size1 = img_size[0]//patch_enc1#only square images
        grid_size2 = grid_size1//patch_enc2
        
        #anchors = {'scale1': [(21, 21)], 'scale2': [(70, 69)]}
        anchors = {'scale1': [(11)], 'scale2': [(35)]}
        #anchors = {'scale1': [(15, 16)], 'scale2': [(27, 27)], 'scale3': [(72, 71)]}
        #anchors = {'scale1': [(9)], 'scale2': [(15)], 'scale3': [(37)]}
    
        final_out_channel = len(anchors[list(anchors.keys())[0]]) * (3 + 1 ) 

        self.conv_final1 = self.make_conv_final(160+n_model//2, final_out_channel) #input is the output of enc2
        #self.detection1 = DetectionRectangle(anchors['scale1'], image_size, 'bbox', 7, device)
        self.detection1 = DetectionCircle(anchors['scale1'], image_size, 'cbbox', 14, device)

        self.upsample = self.make_upsample(640, n_model//2, scale_factor=2)
        self.conv_final2 = self.make_conv_final(128+n_model//2, final_out_channel)

        #self.detection2 = DetectionRectangle(anchors['scale2'], image_size, 'bbox', 14, device)
        self.detection2 = DetectionCircle(anchors['scale2'], image_size, 'cbbox', 28, device)
        
        self.upsample2 = self.make_upsample(160+n_model//2, n_model//2, scale_factor=2)
        #self.conv_final3 = self.make_conv_final(128+n_model//2, final_out_channel)
        #self.detection3 = DetectionRectangle(anchors['scale3'], image_size, 'bbox', 28, device)
        #self.detection3 = DetectionCircle(anchors['scale3'], image_size, 'cbbox', 28, device)

        self.gamma = gamma
        self.alpha = alpha
        self.layers = [self.detection1, self.detection2]
        
    def forward(self, x, targets=None):
        loss=0
        
        # Get backbone features (example shapes)
        features = self.backbone.forward_features(x)  # shape: (3, 197, 192) for TinyViT-5M

        b = features[0].shape[0]
        # Reshape to spatial format
        # Reshape to (batch, channels, height, width)
        shapes = [(28, 28), (14, 14), (7, 7), (7, 7)]
        spatial_features = [ feat.permute(0, 2, 1).reshape(b, -1, *shapes[i])
            for i, feat in enumerate(features)]
        #for i in spatial_features:
        #    print(i.shape)
        
        x = torch.cat((spatial_features[2], spatial_features[3]), dim=1) #b, 640, 7, 7
        x = self.upsample(x) #n_model//2
        x = torch.cat((x, spatial_features[1]), dim=1)#160+n_model//2
        #############################################################################################
        scale1 = self.conv_final1(x)
        #scale1_box = self.conv_final1_box(x)
        #scale1_cls = self.conv_final1_cls(x)
        #scale1 = torch.cat((scale1_box, scale1_cls), dim=1)
        
        output1, layer_loss = self.detection1(scale1, targets, self.gamma, self.alpha)
        #print(targets.shape)
        #print(layer_loss)
        loss += layer_loss
        
        x = self.upsample2(x)
        x = torch.cat((x, spatial_features[0]), dim=1)
        
        scale2 = self.conv_final2(x)
        #scale2_box = self.conv_final2_box(x)
        #scale2_cls = self.conv_final2_cls(x)
        #scale2 = torch.cat((scale2_box, scale2_cls), dim=1)
        #print(targets.shape)
        output2, layer_loss = self.detection2(scale2, targets, self.gamma, self.alpha)
        #print(layer_loss)
        loss += layer_loss
        
        #x = self.upsample2(x)
        #x = torch.cat((x, spatial_features[0]), dim=1)
        #scale3 = self.conv_final3(x)
        
        #output3, layer_loss = self.detection3(scale3, targets, self.gamma, self.alpha)
        #loss += layer_loss

        outputs = torch.cat([output1.detach(), output2.detach()], 1)
        return outputs if targets is None else (loss, outputs)

    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

        modules = nn.Sequential(module1, module2, nn.GELU())#nn.LeakyReLU(negative_slope=0.1))
        return modules

    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            #self.make_conv(in_channels, in_channels//4, kernel_size=3, padding=1),
            #self.make_conv(in_channels//4, in_channels//8, kernel_size=3, padding=1),
            self.make_conv(in_channels, in_channels//4, kernel_size=3),
            self.make_conv(in_channels//4, in_channels//8, kernel_size=3),
            
            #nn.Conv2d(in_channels//8, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return modules

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules

import timm
class PVT_det(nn.Module):
    def __init__(self, image_size=224, n_model=512, device='cpu', gamma=2.0, alpha=0.75):
        super().__init__()
        self.backbone = timm.create_model('pvt_v2_b0', pretrained=True, features_only=True)
        self.backbone.eval()  # Freeze running stats
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        patch_enc1 = 8
        patch_enc2 = 2
        img_size=(image_size, image_size)
        
        grid_size1 = img_size[0]//patch_enc1#only square images
        grid_size2 = grid_size1//patch_enc2
        
        #anchors = {'scale1': [(21, 21)], 'scale2': [(70, 69)]}
        anchors = {'scale1': [(11)], 'scale2': [(35)]}
        final_out_channel = len(anchors[list(anchors.keys())[0]]) * (3 + 1 ) 

        self.conv_final1 = self.make_conv_final(160+256, final_out_channel) #input is the output of enc2

        #self.detection1 = DetectionRectangle(anchors['scale1'], image_size, 'bbox', grid_size2, device)
        self.detection1 = DetectionCircle(anchors['scale1'], image_size, 'cbbox', grid_size2, device)

        self.upsample = self.make_upsample(160+256, n_model//2, scale_factor=2)
        self.conv_final2 = self.make_conv_final(64+64+n_model//2, final_out_channel)

        #self.detection2 = DetectionRectangle(anchors['scale2'], image_size, 'bbox', grid_size1, device)
        self.detection2 = DetectionCircle(anchors['scale2'], image_size, 'cbbox', grid_size1, device)
        
        self.downsample = nn.Conv2d(
                            in_channels=32,       # Number of input channels (adjust as needed)
                            out_channels=64,     # Number of output channels
                            kernel_size=3,       # 3x3 kernel
                            stride=2,            # Critical for downsampling
                            padding=1            # Preserves spatial dimensions before stride
                        )
        self.gamma = gamma
        self.alpha = alpha
        self.layers = [self.detection1, self.detection2]
        
    def forward(self, x, targets=None):
        loss=0
        
        # Get backbone features (example shapes)
        features = self.backbone(x) 
        
        up1 = self.up1(features[3])#14, 14
            
        residual_output = features[1]#b, 64, 28, 28
        x = torch.cat((features[2], up1), dim=1)#160+256
        #############################################################################################
        scale1 = self.conv_final1(x)
        #scale1_box = self.conv_final1_box(x)
        #scale1_cls = self.conv_final1_cls(x)
        #scale1 = torch.cat((scale1_box, scale1_cls), dim=1)
        
        output1, layer_loss = self.detection1(scale1, targets, self.gamma, self.alpha)
        #print(targets.shape)
        #print(layer_loss)
        loss += layer_loss
        
        x = self.upsample(x)
        feat = self.downsample(features[0])
        x = torch.cat((x, residual_output, feat), dim=1)
        
        scale2 = self.conv_final2(x)
        #scale2_box = self.conv_final2_box(x)
        #scale2_cls = self.conv_final2_cls(x)
        #scale2 = torch.cat((scale2_box, scale2_cls), dim=1)
        #print(targets.shape)
        output2, layer_loss = self.detection2(scale2, targets, self.gamma, self.alpha)
        #print(layer_loss)
        loss += layer_loss

        outputs = torch.cat([output1.detach(), output2.detach()], 1)
        return outputs if targets is None else (loss, outputs)

    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)

        modules = nn.Sequential(module1, module2, nn.GELU())#nn.LeakyReLU(negative_slope=0.1))
        return modules

    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            #self.make_conv(in_channels, in_channels//4, kernel_size=3, padding=1),
            #self.make_conv(in_channels//4, in_channels//8, kernel_size=3, padding=1),
            self.make_conv(in_channels, in_channels//4, kernel_size=3),
            self.make_conv(in_channels//4, in_channels//8, kernel_size=3),
            
            #nn.Conv2d(in_channels//8, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return modules

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules
    
if __name__ == '__main__':
    model = TransformerObjectDetection(image_size=224, N_channels=3, n_model=512, num_blks=1, device='cpu')
    print(model)

    test = torch.rand([1, 3, 224, 224])
    y = model(test)
    print(y.shape)
