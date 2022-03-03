# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Modified loss to optimize over Translation rather than R|t
#See also original dense fusion work
# Written by Theo*
# --------------------------------------------------------
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor
import sys


def loss_calculation(pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    knn = KNearestNeighbor(1)
    bs, num_p, _ = pred_c.size()

    print(f'\nthe model point size just inside Loss is : {model_points.size()} and the targt point size is {target.size()} \n')
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    print(f'\nthe model point size after 2 line inside Loss is : {model_points.size()} and the targt point size is {target.size()} \n')
    ori_target = target
    pred_t = pred_t.repeat(1,500,1)
    print(f'pred T es de la forme (avant toutes modif) : {pred_t.size()}\n')
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    print(f'pred T es de la forme (avant contiguous) : {pred_t.size()}\n')
    ori_t = pred_t

    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)
    print(f' the size of points is : {points.size()} \n ')

    print(f'Pred T après ajout de point es de la forme : {pred_t.size()}\n')
    pred = torch.add(model_points , points + pred_t)
    print(f'la prediction final de t es de la forme : {pred.size()}\n')


    # if not refine:
    #     if idx[0].item() in sym_list:
    #         print('ALERT WE ARE IN SYM LOOP OBJECT : \n')
    #         target = target[0].transpose(1, 0).contiguous().view(3, -1)
    #         pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
    #         inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
    #         target = torch.index_select(target, 1, inds.view(-1) - 1)
    #         target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
    #         pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    # cij_model = torch.bmm(pred.transpose(1,0) , pred)
    # cij_Gt = torch.bmm(target.transpose(1,0) , target)

    cij_model = torch.matmul((pred.contiguous()), pred.transpose(2,1))
    cij_Gt = torch.matmul((target.contiguous()), target.transpose(2,1))
    print(f'cij model es : {cij_model} et cij_gt es {cij_Gt}\n et leur taille : cij {cij_model.size()} et gt {cij_Gt.size()}\n')
    # err = torch.mean(torch.norm((cij_model - cij_Gt), dim=2), dim=1)
    err = torch.mean(torch.norm((cij_model - cij_Gt), dim=2), dim=1)
    print(f'L err es {err}\n')
    print(f'la valeur moyenne de lerreur es : {torch.mean(err)}')
    loss = torch.mean((err * pred_c - w * torch.log(pred_c)), dim=0)
    print(f'la loss es {loss}\n')
    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    err = err.view(bs, num_p)

    t = ori_t[which_max[0]] + points[which_max[0]]
    print(f'la valeur de t ayant la plus grande confiance es :{t}\n')
    points = points.view(1, bs * num_p, 3)
    print(f'les valeurs dans points sont tel que : {points} \n ')

    # ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)
    print(f'les valeurs dans ori t sont tel que : {ori_t}\n')
    new_points = (points - ori_t).contiguous()
    print(f'noewpoints correspond  a : {new_points}\n')

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
    new_target = (new_target - ori_t).contiguous()

    sys.exit("\nERROR 106: Program EXIT~System INTERRUPT || Normal SHUTDOWN CODE STOPPED NORMALLY || CHECK Tpredicted format COMPATIBILITY\n ")

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    del knn
    return loss, err[0][which_max[0]], new_points.detach(), new_target.detach()


class LossT(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(LossT, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)


        # for i in model_points :
        #     a = 0
        #     b = a + 1
        #     for j in target:
        #         print(f'The size of i si {i.size()}, the size on j is : {j.size()}')
        #         cij_model = ((i[a] + pred_t).transpose(1,0) * (i[b] + pred_t))
        #         # cij_Gt = ((j[a] + pred_t).transpose(1,0) * (j[b] + pred_t))
        #         a += 1
        #         b += 1
        #         # print(f'the cij model is : {cij_model} and of size : {cij_model.size()}')
        #         # print(f'the cij GT is : {cij_Gt} and of size : {cij_Gt.size()}')
        #         if a == 15:
        #             break
        #     break


# from torch.nn.modules.loss import _Loss
# from torch.autograd import Variable
# import torch
# import time
# import numpy as np
# import torch.nn as nn
# import random
# import torch.backends.cudnn as cudnn
# from lib.knn.__init__ import KNearestNeighbor
#
# def loss_calculation(pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
#     knn = KNearestNeighbor(1)
#     bs, num_p, _ = pred_c.size()
#     print(f'bs is : {bs}, and num_p {num_p}')
#
#     # pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
#
#
#     # base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
#     #                   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
#     #                   (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
#     #                   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
#     #                   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
#     #                   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
#     #                   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
#     #                   (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
#     #                   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
#
#     # ori_base = base
#     # base = base.contiguous().transpose(2, 1).contiguous()
#
#
#     model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
#     target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
#     print(f'the model point size is : {model_points.size()} and the targt point size is {target.size()}')
#
#     ori_target = target
#     pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
#     print(f'the size of pred_t is : {pred_t.size()}')
#     ori_t = pred_t
#     points = points.contiguous().view(bs * num_p, 1, 3)
#     pred_c = pred_c.contiguous().view(bs * num_p)
#
#     pred = torch.add(torch.bmm(model_points, base), points + pred_t)
#     print(f'pred size is : {pred.size()}')
#
#     # for i in model_points :
#     #     for j = i+1 in model_points:
#     #         print(model_points[i],modelpoints[j])
#     #         cij_model = ((model_points[i] + pred_t).transpose(1,0) * (models_points[j] + pred_t))
#     #         cij_Gt = ((target[i] + pred_t).transpose(1,0) * (target[j] + pred_t))
#
#     # for i in model_points :
#     #     for j in model_points:
#     #         print(i.size(),j.size())
#     #
#     #         cij_model = ((model_points[i] + pred_t).transpose(1,0) * (models_points[j] + pred_t))
#     #         cij_Gt = ((target[i] + pred_t).transpose(1,0) * (target[j] + pred_t))
#
#     for i in model_points :
#         a = 0
#         b = a + 1
#         for j in target:
#             print(f'Inside (i a) there is : {i[a]}')
#             print(f'Inside (j a) there is : {j[a]}')
#             print(f'The size of i si {i.size()}, the size on j is : {j.size()}')
#             cij_model = ((i[a] + pred_t).transpose(1,0) * (i[b] + pred_t))
#             cij_Gt = ((j[a] + pred_t).transpose(1,0) * (j[b] + pred_t))
#             a += 1
#             b += 1
#             # print(f'the cij model is : {cij_model} and of size : {cij_model.size()}')
#             # print(f'the cij GT is : {cij_Gt} and of size : {cij_Gt.size()}')
#             break
#         break
#
#     if not refine:
#         if idx[0].item() in sym_list:
#             target = target[0].transpose(1, 0).contiguous().view(3, -1)
#             pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
#             inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
#             target = torch.index_select(target, 1, inds.view(-1) - 1)
#             target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
#             pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
#
#     dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
#     loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)
#
#
#     pred_c = pred_c.view(bs, num_p)
#     how_max, which_max = torch.max(pred_c, 1)
#     dis = dis.view(bs, num_p)
#
#
#     t = ori_t[which_max[0]] + points[which_max[0]]
#     points = points.view(1, bs * num_p, 3)
#
#     # ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
#
#     ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)
#     new_points = torch.bmm((points - ori_t), ori_base).contiguous()
#
#     new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
#     ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
#
#     new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()
#
#     # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
#     del knn
#     return loss, dis[0][which_max[0]], new_points.detach(), new_target.detach()
#
#
# class LossT(_Loss):
#
#     def __init__(self, num_points_mesh, sym_list):
#         super(LossT, self).__init__(True)
#         self.num_pt_mesh = num_points_mesh
#         self.sym_list = sym_list
#
#     def forward(self, pred_t, pred_c, target, model_points, idx, points, w, refine):
#
#         return loss_calculation(pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)
