import numpy as np
import torch
import torch.nn as nn
from glob import glob
import pathlib
import scipy
import os
import scipy.io as scio
import time
import spconv.pytorch as spconv

smax = torch.nn.Softmax2d()

# Process the real UGV-45m data
def test_on_UGV45m(ssunet1, signal_dilation, ssunet2, opt):
    print(opt["testDataRootDir"] + opt["testData"])
    root_path = opt["testDataRootDir"] + opt["testData"]

    outdir_m = opt["testOutDir"] + opt["testData"]

    C = 3e8
    Tp = 100e-12

    h,w = 64,64

    with torch.no_grad():
        for name_test in glob(root_path + "*.mat"):
            name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
            name_test = root_path + name_test_id + ".mat"
            name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"
            print("Loading " + name_test + "......")

            M_mea = scio.loadmat(name_test)["spad"]
            M_mea = scipy.sparse.csc_matrix.todense(M_mea)
            M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, w, h, -1])
            cc = M_mea
            M_mea = np.transpose(M_mea, (0, 4, 3, 2, 1))
            M_mea = torch.from_numpy(M_mea).cuda()

            start_time = time.time()

            # Denoising
            out1 = ssunet1(M_mea)
            dilated_out = signal_dilation(out1)

            # Take the intersection of the raw data and the dilated_out
            dilated_out = dilated_out.permute(0, 2, 3, 4, 1)
            mask = dilated_out > 0
            M_mea = M_mea + 1
            dilated_out = M_mea * mask
            dilated_out = spconv.SparseConvTensor.from_dense(dilated_out)

            # Depth estimating
            out = ssunet2(dilated_out)
            mask = out == 0
            out[mask] = -20
            his = torch.squeeze(out)
            _,out = torch.max(his,dim=0)

            end_time = time.time()
            t_all = end_time - start_time

            print("%f seconds to process %s." % (t_all, name_test))

            scio.savemat(name_test_save, {"img": out.data.cpu().numpy() * Tp * C / 2 + 0.012})
            torch.cuda.empty_cache()

    return 0


# Process the simulated data
def test_on_simulated_data(ssunet1, signal_dilation, ssunet2, opt):
    print(opt["testDataRootDir"] + opt["testData"])
    root_path = opt["testDataRootDir"] + opt["testData"]
    outdir_m = opt["testOutDir"] + opt["testData"]

    sigmoid_func = torch.nn.Sigmoid()

    C = 3e8
    Tp = 80e-12

    with torch.no_grad():
        for name_test in glob(root_path + "*.mat"):
            name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
            name_test = root_path + name_test_id + ".mat"
            name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"
            print("Loading " + name_test + "......")

            dep = np.asarray(scio.loadmat(name_test)['depth']).astype(np.float32)
            h, w = dep.shape
            M_mea = scio.loadmat(name_test)["spad"]
            M_mea = scipy.sparse.csc_matrix.todense(M_mea)
            M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, w, h, -1])
            cc = M_mea
            M_mea = np.transpose(M_mea, (0, 4, 3, 2, 1))
            M_mea = torch.from_numpy(M_mea).cuda()

            start_time = time.time()

            # Denoising
            out1 = ssunet1(M_mea)
            dilated_out = signal_dilation(out1)

            # Take the intersection of the raw data and the dilated_out
            dilated_out = dilated_out.permute(0, 2, 3, 4, 1)
            mask = dilated_out > 0
            M_mea = M_mea + 1
            dilated_out = M_mea * mask
            dilated_out = spconv.SparseConvTensor.from_dense(dilated_out)

            # Depth estimating
            out = ssunet2(dilated_out)
            mask = out == 0
            out[mask] = -20
            his = torch.squeeze(out)

            # # Argmax-based depth estimation
            _, out = torch.max(his, dim=0)

            # # Peak compensation-based depth estimation
            # his = his.unsqueeze(0)
            # his = smax(his)
            # his = torch.squeeze(his)
            # his,out = torch.topk(his,2,dim=0)
            # his = his / his.sum(dim=0,keepdim=True)
            # out = his*out
            # out = out.sum(dim=0)

            end_time = time.time()
            t_all = end_time - start_time

            print("%f seconds to process %s." % (t_all, name_test))

            dist = out.data.cpu().numpy() * Tp * C / 2 + 0.012
            rmse = np.sqrt(np.mean((dist - dep) ** 2))
            print("RMSE: %fm.\n" % (rmse))

            scio.savemat(name_test_save, {"img": dist,"rmse":rmse})
            torch.cuda.empty_cache()

    return 0


