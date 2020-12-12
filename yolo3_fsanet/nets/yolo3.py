import torch
import torch.nn as nn
from collections import OrderedDict
from darknet import darknet53
from FSAnet import FSANet


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out,
                           kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1))
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                  stride=1, padding=0, bias=True)
    ])
    return m


class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet53(None)

        out_filters = self.backbone.layers_out_filters
        #  last_layer0
        final_out_filter0 = len(
            config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.last_layer0 = make_last_layers(
            [512, 1024], out_filters[-1], final_out_filter0)

        #  embedding1
        final_out_filter1 = len(
            config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers(
            [256, 512], out_filters[-2] + 256, final_out_filter1)

        #  embedding2
        final_out_filter2 = len(
            config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"]) # 3*(5+num_class)
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers(
            [128, 256], out_filters[-3] + 128, final_out_filter2)

        # layer for train yaw, pitch, roll
        # input shape bs x [256, 512, 1024] x [(52x52), (26x26), (13x13)]
        # output shape bs x 3 x 3 x [(52x52), (26x26), (13x13)]
        self.num_primcaps = 3*3
        self.primcaps_dim = 8
        self.num_out_capsule = 3
        self.out_capsule_dim = 8
        self.routings = 2
        self.num_anchors = 3
        out_filter_cv = self.num_primcaps*self.primcaps_dim*self.num_anchors
        self.cv2 = conv2d(out_filters[-3], out_filter_cv, 1)
        self.cv1 = conv2d(out_filters[-2], out_filter_cv, 1)
        self.cv0 = conv2d(out_filters[-1], out_filter_cv, 1)
        # self.FSAnet0 = FSANet(self.num_primcaps, self.primcaps_dim, self.num_out_capsule, self.out_capsule_dim, self.routings)
        # self.FSAnet1 = FSANet(self.num_primcaps, self.primcaps_dim, self.num_out_capsule, self.out_capsule_dim, self.routings)
        # self.FSAnet2 = FSANet(self.num_primcaps, self.primcaps_dim, self.num_out_capsule, self.out_capsule_dim, self.routings)

    def forward(self, x):
        def _branch(last_layer, layer_in):
            out_branch = None
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch
        #  backbone
        # train bounding box
        # output shapes: (bsx256x52x52), (bsx512x26x26), (bsx1024x13x13)
        x2, x1, x0 = self.backbone(x)

        # out_branch shape bs*3*(5+num_class)*[(52x52), (26x26), (13x13)]
        #  yolo branch 0
        out0, out0_branch = _branch(self.last_layer0, x0)

        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        #  yolo branch 2
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.last_layer2, x2_in)

        # train 3 head pose: yaw, pitch, roll
        pose0_branch = self.cv0(x0)
        pose1_branch = self.cv1(x1)
        pose2_branch = self.cv2(x2)


        return (out0, pose0_branch), (out1, pose1_branch), (out2, pose2_branch)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append('/media/2tb/Hoang/multitask/code/yolo3_fsanet')
    sys.path.append('/media/2tb/Hoang/multitask/code/yolo3_fsanet/nets')
    os.chdir("/media/2tb/Hoang/multitask/code/yolo3_fsanet/nets")
    from utils.config import Config
    model = YoloBody(Config)
    print(model)
    x = torch.zeros((1, 3, 416, 416))
    y = model(x)
    print(y[0][0].shape)
    print(y[0][1].shape)
