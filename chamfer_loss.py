# https://github.com/lijx10/SO-Net/blob/master/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import faiss
import cfg

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        # self.opt = opt
        self.dimension = 3
        self.k = 1

        # we need only a StandardGpuResources per GPU
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemoryFraction(0.1)
        self.flat_config = faiss.GpuIndexFlatConfig()
        self.flat_config.device = cfg.gpu_id

        # place holder
        self.forward_loss = torch.FloatTensor([0])
        self.backward_loss = torch.FloatTensor([0])

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        # index = faiss.GpuIndexFlatL2(self.res, self.dimension, self.flat_config)  # dimension is 3
        index_cpu = faiss.IndexFlatL2(self.dimension)
        index = faiss.index_cpu_to_gpu(self.res, cfg.gpu_id, index_cpu)
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in GPU
                 I: Variable of Nxk, type LongTensor, in GPU
        '''
        D, I = index.search(query, k)

        D_var =torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))
        # if self.opt.gpu_id >= 0:
        D_var = D_var.to(cfg.DEVICE)
        I_var = I_var.to(cfg.DEVICE)

        return D_var, I_var

    def forward(self, predict_pc, gt_pc):
        '''
        :param predict_pc: Bx3xM Variable in GPU
        :param gt_pc: Bx3xN Variable in GPU
        :return:
        '''

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy())  # BxMx3
        gt_pc_np = np.ascontiguousarray(torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy())  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2])
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2])

        # if self.opt.gpu_id >= 0:
        selected_gt_by_predict = selected_gt_by_predict.to(cfg.DEVICE)
        selected_predict_by_gt = selected_predict_by_gt.to(cfg.DEVICE)

        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_var = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i,k,...] = gt_pc[i].index_select(1, I_var[:,k])

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_var = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i,k,...] = predict_pc[i].index_select(1, I_var[:,k])

        # compute loss ===================================================
        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        forward_loss_element = robust_norm(selected_gt_by_predict-predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict))
        self.forward_loss = forward_loss_element.mean()
        self.forward_loss_array = forward_loss_element.mean(dim=1).mean(dim=1)

        # selected_predict(Bxkx3xN) vs gt_pc(Bx3xN)
        backward_loss_element = robust_norm(selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt))  # BxkxN
        self.backward_loss = backward_loss_element.mean()
        self.backward_loss_array = backward_loss_element.mean(dim=1).mean(dim=1)

        self.loss_array = self.forward_loss_array + self.backward_loss_array
        return self.forward_loss + self.backward_loss # + self.sparsity_loss

    def __call__(self, predict_pc, gt_pc):
        # start_time = time.time()
        loss = self.forward(predict_pc, gt_pc)
        # print(time.time()-start_time)
        return loss