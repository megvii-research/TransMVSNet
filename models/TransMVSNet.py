import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .FMT import FMT_with_pathway

Align_Corners_Range = False

class PixelwiseNet(nn.Module):

    def __init__(self):

        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=1, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1):
        """forward.

        :param x1: [B, 1, D, H, W]
        """

        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1) # [B, D, H, W]
        output = self.output(x1)
        output = torch.max(output, dim=1, keepdim=True)[0] # [B, 1, H ,W]

        return output


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None, view_weights=None):
        """forward.

        :param stage_idx: int, index of stage, [1, 2, 3], stage_1 corresponds to lowest image resolution
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, regularization network
        :param prob_volume_init:
        :param view_weights: pixel wise view weights for src views
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]

        # step 3. cost volume regularization
        cost_reg = cost_regularization(similarity)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
        if view_weights == None:
            view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, view_weights.detach()
        else:
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}


class TransMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
            grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8]):
        super(TransMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
            depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
                "stage1":{
                    "scale": 4.0,
                    },
                "stage2": {
                    "scale": 2.0,
                    },
                "stage3": {
                    "scale": 1.0,
                    }
                }

        self.feature = FeatureNet(base_channels=8)

        self.FMT_with_pathway = FMT_with_pathway()

        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=1, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=1, base_channels=self.cr_base_chs[i])
                for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()

        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrices, depth_values, test_tnt=False):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        features = self.FMT_with_pathway(features)

        outputs = {}
        depth, cur_depth = None, None
        view_weights = None
        for stage_idx in range(self.num_stage):
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            Using_inverse_d = False

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                        [img.shape[2], img.shape[3]], mode='bilinear',
                        align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            # [B, D, H, W]
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                    ndepth=self.ndepths[stage_idx],
                    depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                    dtype=img[0].dtype,
                    device=img[0].device,
                    shape=[img.shape[0], img.shape[2], img.shape[3]],
                    max_depth=depth_max,
                    min_depth=depth_min,
                    use_inverse_depth=Using_inverse_d)

            if stage_idx + 1 > 1: # for stage 2 and 3
                view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")

            if view_weights == None: # stage 1
                outputs_stage, view_weights = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx], view_weights=view_weights)
            else:
                outputs_stage = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx], view_weights=view_weights)

            wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
            depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)
            outputs_stage['depth'] = depth

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs
