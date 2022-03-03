import _init_paths
import argparse
import os
import random
import numpy as np
from numpy.linalg import norm
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys

from torch.autograd import Variable

from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod

from lib.network import PoseNet, PoseRefineNet, PoseNetTranslation
from lib.loss import Loss
from lib.LossTranslation import LossT
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--R_invariant', type=bool, default = False, help='Select the loss type over R invariant or standard DenseFusion' )
opt = parser.parse_args()

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 0
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

if opt.R_invariant == False:
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
else :
    estimator = PoseNetTranslation(num_points = num_points, num_obj = num_objects)
    estimator.cuda()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
# tentative de load pour print out l'estimation de densefusion
comparator = PoseNet(num_points = num_points, num_obj = num_objects)
comparator.cuda()
comparator.load_state_dict(torch.load('/home/tpetitjean/PhD_Thesis/DenseFusion/trained_models/linemod/pose_model_9_0.01310166542980859.pth'))
comparator.eval()

cul = 0
listeu = []
resul = [0,0,0]

refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()

if opt.R_invariant == False:
    criterion = Loss(num_points_mesh, sym_list)
else:
    criterion = LossT(num_points_mesh, sym_list)
    crit_comp = Loss(num_points_mesh, sym_list)

criterion_refine = Loss_refine(num_points_mesh, sym_list)
diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)

for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')
dw = open('{0}/Comparator_result.txt'.format(output_result_dir), 'w')

for i, data in enumerate(testdataloader, 0):

    if cul == 10:
        print(f'the value contain in res 1 : \n {resul} \n and tha value in res 2 :\n {listeu}\n ')
        print(f'the mean the value of re1 is : \n {resul / cul }\n')
        print(f'the mean diffrence between the value of re2 is : \n {np.mean(listeu)}\n')
        # dw.write()
        sys.exit("ERROR 106: Program EXIT~System INTERRUPT || Normal SHUTDOWN CODE STOPPED NORMALLY || CHECK DataSET LOADER incompatibility \n ")

    if len(data) == 6:
        points, choose, img, target, model_points, idx = data
        print(f'---------------------------\nPoint size : {points.size()}\nChooses size : {choose.size()}\nImg size : {img.size()}\nTarget : {target.size()}\nModel point : {model_points.size()}\nIdx : {idx}\n---------------------------\n ')
    else:
        points, __, __, GT_t, GT_r, choose, img, target, model_points, idx = data

    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue

    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()

    if opt.R_invariant == False:
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)
        print(f'après prés traitement on obtiens MtTea de taille {my_t * 1000} et MyAir es : {my_r} et GT_t es : {GT_t} \n  ')

    else:
        # SOA comparator -------------------------------------------------------------------------------------
        pred_r_OriP, pred_t_OriP, pred_c_OriP, emb_OriP = comparator(img, points, choose, idx)
        pred_r_OriP = pred_r_OriP / torch.norm(pred_r_OriP, dim=2).view(1, num_points, 1)
        pred_c_OriP = pred_c_OriP.view(bs, num_points)
        how_max, which_max = torch.max(pred_c_OriP, 1)
        pred_t_OriP = pred_t_OriP.view(bs * num_points, 1, 3)
        DF_r = pred_r_OriP[0][which_max[0]].view(-1).cpu().data.numpy()
        DF_t = (points.view(bs * num_points, 1, 3) + pred_t_OriP)[which_max[0]].view(-1).cpu().data.numpy()
        DF_pred = np.append(DF_r, DF_t)
        #-----------------------------------------------------------------------------------------------------
        pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_r = [1,0,0,0]
# _____________________________________________________________________________________________________________________________________.
#                                                 ENTER IN TEST | CALCULATION ZONE                                                     |
# _____________________________________________________________________________________________________________________________________|
        cul += 1

        my_tea = my_t * 1000
        mighty = GT_t           # sous format torch
        my_t_gt = GT_t.numpy()  # sous format numpy

        GtR = GT_r.numpy()     # compute a quaternion de taille correct [1,4]

        v2add = np.array([0,0,0])
        v2add2 = np.array([0,0,0,1])
        res = np.hstack((np.reshape(GtR, (3,3)), np.atleast_2d(v2add).T))
        rezu = np.vstack((res, v2add2))
        R_GT = quaternion_from_matrix(rezu, True)     # on mets la matrice R_GT sous forme de quaternion en injectant l'identité sur la 4 emem ligne et colone

        my_pred = np.append( R_GT , GT_t)

        up = my_tea * my_t_gt
        down = norm(my_tea ,2) * norm(my_t_gt,2)
        res = up / down

        res4 = np.linalg.norm(my_tea) - np.linalg.norm(my_t_gt)

        resul = resul + res
        listeu.append(abs(res4))

        dw.write('----------------------\n No.{0} : \n The Gt is : t{1} \nand R{2}\n The DenseFusion prediction are :t_DF{3} and R_DF{4}\n The calculation resulat are : {5}, {6} \n And the initial prediction before refine is {7} \n'.format(cul, my_t_gt,GtR,DF_t,DF_r,res,res4,my_tea))
        print(f'----------------------\n iteration numéro {cul}\n \n la Gt de T es :{my_t_gt / 1000 }\n la GT de R es :\n{GtR} \n ')
        print(f'The Dense Fusion prediction is : {DF_pred}\n with R = {DF_r} and t = {DF_t} \n')
        print(f'La prediction de notre réseau rotation invariant es : {my_t}\n')
        print(f'Le resultats du calcul es comme suit (calcul effectuer sous torch): \n {res}\n et le résultats du second calcul es : \n {res4}\n')
        # print(f'La prediction initial de t par le réseau avant refine: \n {my_tea} \n ')
# ------------------------------------------------------------------------------------------------------------------------------------ #

    # print(f'le resultats du calcul es comme suit : \n {res}\n\n et le résultats du second calcul es : \n {res2}\n')
    # print(f'----------------------\n My_t es : \n {my_tea*1000}\n\n La valeur de Gt_t es : \n{GT_t}\n\n RGT es de la forme : \n{R_GT} \n----------------------\n')
    # print(f'les valeurs de x y et z sont {dir}')
    # print(f'dans eval on resize T avec bs {bs} num_point {num_points} puis pred_t après traitement {pred_t.size()} \n')
    # print(f'on as aussi how max {how_max} de traille : {how_max.size()}, et wich_max {which_max} de taille {which_max.size()}')

    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(GtR)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)

        my_mat[0:3, 3] = my_t
        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final
        # print(f'----------------------\nLa prediction après refine es My_t es :\n {my_t_final}\n\n La valeur de Gt_t es :\n {GT_t}\n \n La valeur de R es : \n {my_r} \n \nLa valeur de GT_R : \n {R_GT}\n\n My pred es :\n {my_pred}\n---------------------- \n')

    model_points = model_points[0].cpu().detach().numpy()
    my_r = quaternion_matrix(R_GT)[:3, :3]
    pred = np.dot(model_points, my_r.T) + my_t
    target = target[0].cpu().detach().numpy()
    # print(f'La prediction Final es My_t es :\n {my_t}\n\n La valeur de R es : \n {my_r} \n \n My pred es :\n {my_pred}\n---------------------- \n')
    # sys.exit("ERROR 106: Program EXIT~System INTERRUPT || Normal SHUTDOWN CODE STOPPED NORMALLY || CHECK DataSET LOADER incompatibility \n ")

    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))

    num_count[idx[0].item()] += 1

for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()
