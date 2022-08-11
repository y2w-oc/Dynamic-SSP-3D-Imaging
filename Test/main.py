# The test file for network
import numpy as np
import torch
import torch.nn as nn
import os
import sys
from glob import glob
import pathlib

from pro.Fn_Test import test_on_UGV45m, test_on_simulated_data
from models import SSPINet
from util.ParseArgs import parse_args


def main():
    # parse arguments
    opt = parse_args("./config.ini")
    print("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
                                                   torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Test Data: {}".format(opt["testData"]))
    print("Test Output Path: {}".format(opt["testOutDir"]+opt["testData"]))


    # list all the pretrained models and configure network
    ssunet1 = SSPINet.SSUnetDenoiser()
    ssunet1.cuda()
    ssunet1.eval()

    signal_dilation = SSPINet.SignalDilation()
    signal_dilation.cuda()
    signal_dilation.eval()

    ssunet2 = SSPINet.SSUnetDepthEstimator()
    ssunet2.cuda()
    ssunet2.eval()

    # *******************************************************************************************
    outdir_m = opt["testOutDir"] + opt["testData"]
    pathlib.Path(outdir_m).mkdir(parents=True, exist_ok=True)

    ckpt = torch.load("./preTrain/ssunet1.pth")
    model_dict = ssunet1.state_dict()

    try:
        ckpt_dict = ckpt["state_dict"]
    except KeyError:
        print('Key error loading state_dict from checkpoint; assuming checkpoint contains only the state_dict')
        ckpt_dict = ckpt

    # to update the model using the pretrained models
    for key_iter, k in enumerate(ckpt_dict.keys()):
        model_dict.update({k: ckpt_dict[k]})
        if key_iter == (len(ckpt_dict.keys()) - 1):
            print('Model Parameter Update!')

    ssunet1.load_state_dict(model_dict)

    # **************************************************************************************************************
    ckpt = torch.load("./preTrain/ssunet2.pth")
    model_dict = ssunet2.state_dict()

    try:
        ckpt_dict = ckpt["state_dict"]
    except KeyError:
        print('Key error loading state_dict from checkpoint; assuming checkpoint contains only the state_dict')
        ckpt_dict = ckpt

    # to update the model using the pretrained models
    for key_iter, k in enumerate(ckpt_dict.keys()):
        model_dict.update({k: ckpt_dict[k]})
        if key_iter == (len(ckpt_dict.keys()) - 1):
            print('Model Parameter Update!')

    ssunet2.load_state_dict(model_dict)

    # run processing function
    if opt["testData"] == "UGV-45m/":
        test_on_UGV45m(ssunet1, signal_dilation, ssunet2, opt)
    else:
        test_on_simulated_data(ssunet1, signal_dilation, ssunet2, opt)


if __name__ == "__main__":
    print("Start testing......")
    main()



