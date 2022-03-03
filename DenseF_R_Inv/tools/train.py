# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# partially modified by Theo*
# --------------------------------------------------------
import _init_paths
import argparse
import os
import sys
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet, PoseNetTranslation
from tensorboardX import SummaryWriter

from lib.loss import Loss
from lib.LossTranslation import LossT
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 50, help='batch size')
parser.add_argument('--workers', type=int, default = 20, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.058, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.033, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 4, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=10, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
parser.add_argument('--R_invariant', type=bool, default = False, help='Select the loss type over R invariant or standard DenseFusion' )
opt = parser.parse_args()

writer = SummaryWriter()

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    print("\nFirst Check Point reeched, we are just before the FIrst load of the NetWork")
    print('-------------------------------------------------------------------------------  \n ')
    stupidcounter = 0
    if opt.R_invariant == False:
        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        estimator.cuda()
    else:
        estimator = PoseNetTranslation(num_points = opt.num_points, num_obj = opt.num_objects)
        estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    print('training Set Loaded ! \n')

    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    print('Testing Set Loaded ! \n ')

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3} \n'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    # --------------------------------------------------------------------------#
    #                               TensorBard Loader
    # dataiter = iter(dataloader)
    # cloud, img_rgb, img_depth, choose, maskimg, target, model, _ = dataiter.next()
    # img_rgb = img_rgb.permute(0,3,2,1)
    # # create grid of images
    # img_grid = torchvision.utils.make_grid(img_depth)
    # img_grid2 = torchvision.utils.make_grid(img_rgb)
    # # show images
    # matplotlib_imshow(img_grid, one_channel=False)
    # matplotlib_imshow(img_grid2, one_channel=False)
    # # write to tensorboard
    # writer.add_image('Depth image exemple ', img_grid)
    # writer.add_image('Image Rgb exemple ', img_grid2)
    # --------------------------------------------------------------------------#

    if opt.R_invariant == False:
        criterion = Loss(opt.num_points_mesh, opt.sym_list)
    else:
        criterion = LossT(opt.num_points_mesh, opt.sym_list)

    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)
    best_test = np.Inf

    print('We hit CP 2, loss, tensorboard and data has been loaded for a first time, now its time to init the epoch ')
    print('-------------------------------------------------------------------------------------- \n ')
    LoopCounter = 0
    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):

            for i, data in enumerate(dataloader, 0):
                LoopCounter = LoopCounter + 1

                points, __, __, ori_t, ori_r, choose, img, target, model_points, idx = data

                points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
                # print(f'What is inside data : points are {points.size()},\n choose is  {choose.size()}, \n image is {img.size()}, \n target is {target.size()},\n model point is {model_points.size()},\n index is {idx.size()}')
                # sys.exit("ERROR 106: Program EXIT~System INTERRUPT || Normal SHUTDOWN CODE STOPPED NORMALLY || CHECK DataSET LOADER incompatibility \n ")

                if opt.R_invariant == False:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                    # print(f'the output of the estimator is : pred R {pred_r.size()}, pred t {pred_t.size()}, pred c {pred_c.size()} and emb {emb.size()}')
                    loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                else:
                    pred_t, pred_c, emb = estimator(img, points, choose, idx)
                    r_gt = ori_r
                    print(r_gt)
                    pred_t = ori_t / 1000
                    pred_t = Variable(pred_t).cuda()
                    print(f'\nla gt de t es : {ori_t / 1000 }\n, et pred original de t {pred_t}\n')
                    loss, dis, new_points, new_target = criterion(pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                    # print(f'the output of the estimator is : pred t {pred_t.size()}, pred c {pred_c.size()}, et idx es : {idx}')
                    # sys.exit("ERROR 106: Program EXIT~System INTERRUPT || Normal SHUTDOWN CODE STOPPED NORMALLY || CHECK DataSET LOADER incompatibility \n ")

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        stupidcounter += 1
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                else:
                    # print('cp3.1 entré dans la loop loss.backward \n ')
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                # ...log the running loss
                writer.add_scalar('training loss', loss / 1000, int(LoopCounter/100))#LoopCounter)
                writer.add_scalar('Mean distance computed',dis , int(LoopCounter/100) )

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4} RepeatEpoch:{5}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size, LoopCounter/100))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, __, __, ori_t, ori_r, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()

            if opt.R_invariant == False:
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
            else:
                pred_t, pred_c, emb = estimator(img, points, choose, idx)
                _, dis, new_points, new_target = criterion(pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))

        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                if opt.R_invariant == False:
                    torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
                else:
                    torch.save(estimator.state_dict(), '{0}/poseRinvar_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            print('is this condition is true somhow ')
            print(stupidcounter)
            # sys.exit("ERROR 119: Program EXIT~System INTERRUPT || Normal SHUTDOWN CODE STOPPED NORMALLY || DEBUGG MODE activated REFINER INC \n ")
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            if opt.R_invariant == False:
                criterion = Loss(opt.num_points_mesh, opt.sym_list)
            else:
                criterion = LossT(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

if __name__ == '__main__':
    main()
