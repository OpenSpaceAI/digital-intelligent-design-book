from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap, img_plot
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy
from lib.train.admin import multigpu

class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.id = 0

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1
        n, b, c, hs, ws = data['search_images'].shape

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        #######################################可视化
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        search_image = np.uint8((search_img[0].permute(1,2,0).detach().cpu()*std + mean).numpy()*255)
        # search_image_v = cv2.resize(search_image, (256, 256))
        # n, b, c, hs, ws = data['search_images'].shape
        #########可视化search_mask
        # search_anno = data['search_anno'].reshape(n * b, -1)
        # search_mask = self.anno2mask(search_anno, ws)
        # plt.imshow(search_mask[0].reshape(256, 256).detach().cpu().numpy())
        # plt.show()
        #######################################
        #######################可视化image+bbox
        # search_anno_box = (data['search_anno'].reshape(n * b, -1)[0].detach().cpu() * 256).numpy()
        # x, y, w, h = search_anno_box
        # search_anno_image = cv2.rectangle(deepcopy(search_image_v), (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (255, 0, 0))
        # plt.imshow(search_anno_image)
        # plt.show()
        # plt.imsave(f'./debug/gradient/wo gt/mean/imageanno_{self.id}.png',search_anno_image)
        ######################################
        #img_plot(search_img[0].cpu())
        #logit = self.importance(data, out_dict, search_image)



        grad = self.importance(data, out_dict, search_image)
        if multigpu.is_multi_gpu(self.net):
            x, aux_dict = self.net.module.backbone(z=template_list,
                                                   x=search_img,
                                                   ce_template_mask=box_mask_z,
                                                   ce_keep_rate=ce_keep_rate,
                                                   return_last_attn=False)
        else:
            x, aux_dict = self.net.backbone(z=template_list,
                                            x=search_img,
                                            ce_template_mask=box_mask_z,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=False)
        search = x[:, -256:]
        # plt.imshow(search[0].mean(-1).reshape(16,16).detach().cpu().numpy())
        # plt.show()
        #search = search * grad
        # plt.imshow(search[0].mean(-1).reshape(16,16).detach().cpu().numpy())
        # plt.show()
        x[:, -256:] = search
        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        if multigpu.is_multi_gpu(self.net):
            out = self.net.module.forward_head(feat_last, None)
        else:
            out = self.net.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        #score = out_dict['score_map'].reshape(n * b, -1)
        # self.net.attentions.clear()
        return out
        #return out_dict

    def anno2mask(self, gt_bboxes, size):
        bboxes = box_xywh_to_xyxy(gt_bboxes) * size # b, 4
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1).cuda() + 0.5 #b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask)

        # cx = torch.floor((bboxes[:, 0]) + bboxes[:, 2] / 2).long()
        # cy = torch.floor((bboxes[:, 1]) + bboxes[:, 3] / 2).long()
        # bid = torch.arange(cx.shape[0]).to(cx)
        # mask[bid, cy, cx] = True

        return mask.flatten(1)

    def importance(self, gt_dict, out_dict, search_image):
    #def importance(self, gt_dict, out_dict):
        n, b, c, hs, ws = gt_dict['search_images'].shape
        search_anno = gt_dict['search_anno'].reshape(n*b, -1)
        search_mask = self.anno2mask(search_anno, ws//16)
        score = out_dict['score_map'].reshape(n*b, -1)
        logit = (score * search_mask).sum(-1).mean()
        #logit = score.sum(-1).mean()
        if multigpu.is_multi_gpu(self.net):
            self.net.module.gradients.clear()
        else:
            self.net.gradients.clear()
        self.net.zero_grad()
        #logit.requires_grad_()
        logit.backward()
        self.net.zero_grad()
        #attns, grads = self.net.attentions, self.net.gradients
        if multigpu.is_multi_gpu(self.net):
            grads = self.net.module.gradients
        else:
            grads = self.net.gradients
        #b, C, sx, sx = grads[0].shape
        #grad = grads[0].reshape(b, C, -1).permute(0, 2, 1)
        _, l, _ = grads[0].shape
        #grad = self.get_first_comp(grads[0][:, l - ws:])
        grad = grads[0][:, l - ws:].mean(-1)[0]
        #grad = grads[0][:, l - ws:]
        #grad = grads[0]

        # max = torch.max(grad, -1, keepdim=True)[0].data
        # min = torch.min(grad, -1, keepdim=True)[0].data
        # grad = (grad - min) / (max - min) #最大最小归一化
        # grad = grad.mean(-1)
        # grad = grad[0]
        with torch.no_grad():
            g_vis = cv2.resize(grad.reshape(16, 16).cpu().numpy(), (256, 256))
            plt.imshow(g_vis)
            plt.show()
            # plt.imsave(f'./debug/gradient/wo gt/mean/gradient_{self.id}.png', g_vis)
            max = np.max(g_vis)
            min = np.min(g_vis)
            g_vis = (g_vis - min) / (max - min)
            #g_vis = g_vis / m
            g_vis = np.uint8(255 * g_vis)
            heatmap = cv2.applyColorMap(g_vis, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            search_image_logit = cv2.addWeighted(search_image, 0.5, heatmap, 0.5, 0.0)
            plt.imshow(search_image_logit)
            plt.show()
            # plt.imsave(f'./debug/gradient/wo gt/mean/image_gradient_{self.id}.png', search_image_logit)
            self.id += 1

        #del self.net.attentions, self.net.gradients
        #self.net.attentions.clear()
        if multigpu.is_multi_gpu(self.net):
            self.net.module.gradients.clear()
        else:
            self.net.gradients.clear()
        #self.net.hooks.clear()
        # for hook in self.net.hooks:
        #     hook.remove()

        return grad

    def get_first_comp(self, x):
        x[torch.isnan(x)] = 0
        x = x - x.mean(1, keepdim=True)
        U, S, V = torch.svd(x, some=False) #torch.svd返回的是V，不是V的转置矩阵
        proj = x @ V[:, :, :1]
        return proj.squeeze()

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
