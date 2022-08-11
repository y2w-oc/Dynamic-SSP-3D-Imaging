import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
import functools

dtype = torch.cuda.FloatTensor


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None, ks=3, pd=1):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels,  kernel_size=ks, padding=pd, bias=True, indice_key=indice_key)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=ks, padding=pd, bias=True, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=ks, padding=pd, bias=True, indice_key=indice_key)
        )

    def forward(self, input):

        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(input).features)

        return output


class SSUnetDenoiser(nn.Module):
    def __init__(self):
        super(SSUnetDenoiser, self).__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        channel_m = 8

        self.start_head = spconv.SparseSequential(
            spconv.SubMConv3d(1, 4, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(4),
            nn.ReLU(),
            spconv.SubMConv3d(4, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
        )

        self.path0_0 = spconv.SparseSequential(
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1),
            # ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1)
        )

        self.downsample0 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample0")
        )

        self.path1_0 = spconv.SparseSequential(
            ResidualBlock(channel_m,channel_m,norm_fn,indice_key="subm1", ks=3, pd=1),
            # ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm1", ks=3, pd=1)
        )

        self.downsample1 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample1")
        )

        self.path2_0 = spconv.SparseSequential(
            ResidualBlock(channel_m,channel_m,norm_fn,indice_key="subm2", ks=3, pd=1),
            # ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm2", ks=3, pd=1)
        )

        self.downsample2 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample2")
        )

        self.path3 = spconv.SparseSequential(
            ResidualBlock(channel_m,channel_m,norm_fn,indice_key="subm3", ks=3, pd=1),
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm3", ks=3, pd=1)
        )

        self.upsample2 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample2")
        )

        self.path2_1 = spconv.SparseSequential(
            ResidualBlock(channel_m*2, channel_m, norm_fn, indice_key="subm2", ks=3, pd=1)
        )

        self.upsample1 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample1")
        )

        self.path1_1 = spconv.SparseSequential(
            ResidualBlock(channel_m*2, channel_m, norm_fn, indice_key="subm1", ks=3, pd=1)
        )

        self.upsample0 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample0")
        )

        self.path0_1 = spconv.SparseSequential(
            ResidualBlock(channel_m*2, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1),
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1)
        )

        self.end_tail = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, 4, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(4),
            nn.ReLU(),
            spconv.SubMConv3d(4, 1, kernel_size=3, padding=1, bias=True, indice_key="subm0"),
            # spconv.ToDense()
        )

    def forward(self, inputs):

        inputs_sp = spconv.SparseConvTensor.from_dense(inputs)

        start_out = self.start_head(inputs_sp)

        out0 = self.path0_0(start_out)

        ds_out0 = self.downsample0(out0)

        out1 = self.path1_0(ds_out0)

        ds_out1 = self.downsample1(out1)

        out2 = self.path2_0(ds_out1)

        ds_out2 = self.downsample2(out2)

        out3 = self.path3(ds_out2)

        up_out3 = self.upsample2(out3)

        out2 = out2.replace_feature(torch.cat((out2.features, up_out3.features), dim=1))

        out2 = self.path2_1(out2)

        up_out2 = self.upsample1(out2)

        out1 = out1.replace_feature(torch.cat((out1.features, up_out2.features), dim=1))

        out1 = self.path1_1(out1)

        up_out1 = self.upsample0(out1)

        out0 = out0.replace_feature(torch.cat((out0.features, up_out1.features), dim=1))

        out0 = self.path0_1(out0)

        out = self.end_tail(out0)

        spatial_shape = out.spatial_shape
        batch_size = out.batch_size
        mask = out.features > 0
        features_th = torch.masked_select(inputs_sp.features, mask)
        features_th = features_th.view(-1, 1)
        indices_th = torch.masked_select(inputs_sp.indices, mask)
        indices_th = indices_th.view(-1, 4)
        del out
        out = spconv.SparseConvTensor(features_th, indices_th, spatial_shape, batch_size)

        return out


class SignalDilation(nn.Module):
    def __init__(self):
        super(SignalDilation, self).__init__()

        self.point_expansion = spconv.SparseSequential(
            spconv.SparseConv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0), bias=False, indice_key="point1"),
            spconv.SparseConv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0), bias=False, indice_key="point2"),
            spconv.SparseConv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0), bias=False, indice_key="point3"),
            # spconv.SparseConv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0), bias=False, indice_key="point4"),
            # spconv.SparseConv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0), bias=False, indice_key="point5"),
            # spconv.SparseConv3d(1, 1, kernel_size=(3,1,1), padding=(1,0,0), bias=False, indice_key="point6"),
        )
        init.constant_(self.point_expansion[0].weight, 1.0)
        init.constant_(self.point_expansion[1].weight, 1.0)
        init.constant_(self.point_expansion[2].weight, 1.0)
        # init.constant_(self.point_expansion[3].weight, 1.0)
        # init.constant_(self.point_expansion[4].weight, 1.0)
        # init.constant_(self.point_expansion[5].weight, 1.0)

        self.plane_expansion = spconv.SparseSequential(
            spconv.SparseConv3d(1, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False, indice_key="plane1"),
            spconv.SparseConv3d(1, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False, indice_key="plane2"),
            spconv.SparseConv3d(1, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False, indice_key="plane3"),
            spconv.SparseConv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, indice_key="plane4"),
            # spconv.SparseConv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, indice_key="plane5"),
            # spconv.SparseConv3d(1, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, indice_key="plane6"),
            spconv.ToDense()
        )
        init.constant_(self.plane_expansion[0].weight, 1.0)
        init.constant_(self.plane_expansion[1].weight, 1.0)
        init.constant_(self.plane_expansion[2].weight, 1.0)
        init.constant_(self.plane_expansion[3].weight, 1.0)
        # init.constant_(self.plane_expansion[4].weight, 1.0)
        # init.constant_(self.plane_expansion[5].weight, 1.0)


    def forward(self, inputs):  # inputs : SparseConvTensor

        point_eout = self.point_expansion(inputs)
        plane_eout = self.plane_expansion(point_eout)

        return plane_eout


# *********************************************************************
class SSUnetDepthEstimator(nn.Module):
    def __init__(self):
        super(SSUnetDepthEstimator, self).__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        channel_m = 8

        self.start_head = spconv.SparseSequential(
            spconv.SubMConv3d(1, 4, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(4),
            nn.ReLU(),
            spconv.SubMConv3d(4, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
        )

        self.path0_0 = spconv.SparseSequential(
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1),
            # ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1)
        )

        self.downsample0 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample0")
        )

        self.path1_0 = spconv.SparseSequential(
            ResidualBlock(channel_m,channel_m,norm_fn,indice_key="subm1", ks=3, pd=1),
            # ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm1", ks=3, pd=1)
        )

        self.downsample1 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample1")
        )

        self.path2_0 = spconv.SparseSequential(
            ResidualBlock(channel_m,channel_m,norm_fn,indice_key="subm2", ks=3, pd=1),
            # ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm2", ks=3, pd=1)
        )

        self.downsample2 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseConv3d(channel_m, channel_m, 3, 2, 1, indice_key="sample2")
        )

        self.path3 = spconv.SparseSequential(
            ResidualBlock(channel_m,channel_m,norm_fn,indice_key="subm3", ks=3, pd=1),
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm3", ks=3, pd=1)
        )

        self.upsample2 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample2")
        )

        self.path2_1 = spconv.SparseSequential(
            ResidualBlock(channel_m*2, channel_m, norm_fn, indice_key="subm2", ks=3, pd=1)
        )

        self.upsample1 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample1")
        )

        self.path1_1 = spconv.SparseSequential(
            ResidualBlock(channel_m*2, channel_m, norm_fn, indice_key="subm1", ks=3, pd=1)
        )

        self.upsample0 = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SparseInverseConv3d(channel_m, channel_m, 3, indice_key="sample0")
        )

        self.path0_1 = spconv.SparseSequential(
            ResidualBlock(channel_m*2, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1),
            ResidualBlock(channel_m, channel_m, norm_fn, indice_key="subm0", ks=3, pd=1)
        )

        self.end_tail = spconv.SparseSequential(
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, channel_m, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(channel_m),
            nn.ReLU(),
            spconv.SubMConv3d(channel_m, 4, kernel_size=3, padding=1, bias=False, indice_key="subm0"),
            norm_fn(4),
            nn.ReLU(),
            spconv.SubMConv3d(4, 1, kernel_size=3, padding=1, bias=True, indice_key="subm0"),
            spconv.ToDense()
        )

    def forward(self, inputs):
        inputs = inputs.replace_feature(inputs.features-1)

        start_out = self.start_head(inputs)

        out0 = self.path0_0(start_out)

        ds_out0 = self.downsample0(out0)

        out1 = self.path1_0(ds_out0)

        ds_out1 = self.downsample1(out1)

        out2 = self.path2_0(ds_out1)

        ds_out2 = self.downsample2(out2)

        out3 = self.path3(ds_out2)

        up_out3 = self.upsample2(out3)

        out2 = out2.replace_feature(torch.cat((out2.features, up_out3.features), dim=1))

        out2 = self.path2_1(out2)

        up_out2 = self.upsample1(out2)

        out1 = out1.replace_feature(torch.cat((out1.features, up_out2.features), dim=1))

        out1 = self.path1_1(out1)

        up_out1 = self.upsample0(out1)

        out0 = out0.replace_feature(torch.cat((out0.features, up_out1.features), dim=1))

        out0 = self.path0_1(out0)

        out = self.end_tail(out0)

        return out
