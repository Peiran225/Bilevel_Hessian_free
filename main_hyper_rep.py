import os
import time
import pdb
import logging
import json
import argparse
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from utils_hyper import *
from dataset import *

import pdb

def finetunning(args, model, x_spt, y_spt, x_qry, y_qry):
    """
    :param x_spt:   [setsz, c_, h, w]
    :param y_spt:   [setsz]
    :param x_qry:   [querysz, c_, h, w]
    :param y_qry:   [querysz]
    :return:
    """
    assert len(x_spt.shape) == 4

    querysz = x_qry.size(0)

    losses_q = [0 for _ in range(args.innerT + 1)] 
    corrects = [0 for _ in range(args.innerT + 1)]

    # in order to not ruin the state of running_mean/variance and bn_weight/bias
    # we finetunning on the copied model instead of self.net
    net = deepcopy(model)
    # pdb.set_trace()
    # 1. run the i-th task and compute loss for k=0
    logits = net(x_spt)
    loss = F.cross_entropy(logits, y_spt)
    grad = torch.autograd.grad(loss, net.getInner_params())
    fast_weights = list(map(lambda p: p[1] - args.lr * p[0], zip(grad, net.getInner_params())))

    # this is the loss and accuracy before first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = net(x_qry, net.parameters(), bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[0] += loss_q
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[0] = corrects[0] + correct

    # this is the loss and accuracy after the first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = net(x_qry, fast_weights, bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[1] += loss_q
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        # scalar
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[1] = corrects[1] + correct

    for k in range(1, args.innerT):
        # 1. run the i-th task and compute loss for k=1~K-1
        logits = net(x_spt, fast_weights, bn_training=True)
        loss = F.cross_entropy(logits, y_spt)
        # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, fast_weights)
        # 3. theta_pi = theta_pi - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - args.lr * p[0], zip(grad, fast_weights)))

        logits_q = net(x_qry, fast_weights, bn_training=True)
        # loss_q will be overwritten and just keep the loss_q on last update step.
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[k + 1] += loss_q

        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
            corrects[k + 1] = corrects[k + 1] + correct


    del net

    accs = np.array(corrects) / querysz
    losses = np.array([l.data.cpu().numpy().item() for l in losses_q])

    return accs[-1], losses[-1]

def train(args, data_loader, logger):     

    eps = 1e-6

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    #inner variable
    model = Learner(config).cuda()
    model_old = Learner(config).cuda()
    assert len(model.parameters()) == 18

    for p, p_old in zip(model.parameters(), model_old.parameters()):
        p_old.data = p.data

    hyperDim = 0
    for p in model.getHyperRep_params():
        hyperDim += len(p.reshape(-1))
    
    innerDim = 0
    for p in model.getInner_params():
        innerDim += len(p.reshape(-1))

    print(hyperDim, innerDim)

    if args.outer_opt == 'SGD':
        opt_lamda = optim.SGD(model.getHyperRep_params(), lr=args.hlr)
    elif args.outer_opt == 'Adam':
        opt_lamda = optim.Adam(model.getHyperRep_params(), lr=args.hlr)

    ###########################Main training loop#################################################
    hyt = 0
    # pdb.set_trace()
    while True:
        for batch, spider_batch in zip(data_loader.dataloader, data_loader.spider_dataloader):
            x_spt, y_spt = batch['train']
            x_qry, y_qry = batch['test']
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

            start_time = time.time()
            if args.alg == 'reverse':
                tmp_grad = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                for t_num in range(args.task_num):
                    new_params = [p for p in model.getInner_params()]
                    for _ in range(args.innerT):
                        logits = model(x_spt[t_num], new_params, bn_training=True)
                        loss = F.cross_entropy(logits, y_spt[t_num])
                        grad = torch.autograd.grad(loss, new_params, create_graph=True)
                        new_params = list(map(lambda p: p[1] - args.lr * p[0], zip(grad, new_params)))

                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    loss = F.cross_entropy(logits_q, y_qry[t_num])
                    p_norm = 0
                    for pp in model.getHyperRep_params():
                        p_norm = p_norm + torch.norm(pp, p=1)
                    if hyt % args.interval  == 0:logger.update_err(loss.data.cpu().numpy().item())
                    grad = torch.autograd.grad(loss + args.l1_alpha * p_norm, model.getHyperRep_params())
                    tmp_grad = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_grad, grad)]

                
                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), tmp_grad):
                    p.grad = g.detach().clone()
                opt_lamda.step()


            elif args.alg == 'AID_CG': 
                tmp_grad = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                for t_num in range(args.task_num):                           
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, grad = hyper_grad_cg(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                    if hyt % args.interval  == 0:logger.update_err(loss)
                    tmp_grad = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_grad, grad)]

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), tmp_grad):
                    p.grad = g.detach().clone()
                opt_lamda.step()
                

            elif args.alg == 'AID_NS':  # or stocBio if without warm start/ or AID_FP / or BSA
                tmp_grad = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                for t_num in range(args.task_num):                           
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                    if hyt % args.interval == 0: logger.update_err(loss)
                    tmp_grad = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_grad, grad)]

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), tmp_grad):
                    p.grad = g.detach().clone()
                opt_lamda.step()

            elif args.alg == 'Dire': 
                tmp_grad = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                tmp_loss = []

                for t_num in range(args.task_num):    
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, grad = hyper_grad_dir(model, new_params, x_spt, y_spt, t_num)

                    tmp_grad = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_grad, grad)]
                    tmp_loss.append(loss)
                
                if hyt % args.interval  == 0:
                    logger.update_err(np.mean(tmp_loss))

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), tmp_grad):
                    p.grad = g.detach().clone()
                opt_lamda.step()


            elif args.alg == 'FSLA': 
                if hyt == 0:
                    v_state = []
                    for p in model.getInner_params():
                        v_state.append(torch.zeros_like(p).cuda())

                tmp_grad = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                tmp_v_state = [torch.zeros_like(p).cuda() for p in model.getInner_params()]
                tmp_v_norm = [0 for _ in np.arange(len(model.getInner_params()))]
                tmp_loss = []

                for t_num in range(args.task_num):    
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, grad, v_s, v_norm = hyper_grad_fsla(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num, v_state)

                    tmp_grad = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_grad, grad)]
                    tmp_v_state = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_v_state, v_s)]
                    tmp_v_norm = [tmp_g + fast_g/args.task_num for tmp_g, fast_g in zip(tmp_v_norm, v_norm)]
                    tmp_loss.append(loss)

                v_state = tmp_v_state

                if hyt % args.interval  == 0:
                    logger.update_err(np.mean(tmp_loss))
                    logger.update_v_norm(tmp_v_norm)


                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), tmp_grad):
                    p.grad = g.detach().clone()
                opt_lamda.step()


            elif args.alg == 'FSLA_ADA': 
                if hyt == 0:
                    v_state = []; v_momentum = []; v_adaptive = []
                    for p in model.getInner_params():
                        v_state.append(torch.zeros_like(p).cuda())
                        v_momentum.append(torch.zeros_like(p).cuda())
                        v_adaptive.append(torch.zeros_like(p).cuda())

                tmp_grad = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                tmp_v_state = [torch.zeros_like(p).cuda() for p in model.getInner_params()]
                tmp_v_momentum = [torch.zeros_like(p).cuda() for p in model.getInner_params()]
                tmp_v_adaptive = [torch.zeros_like(p).cuda() for p in model.getInner_params()]

                tmp_v_norm = [0 for _ in np.arange(len(model.getInner_params()))]
                tmp_loss = []

                for t_num in range(args.task_num):    
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, grad, v_s, v_m, v_ada, v_norm = hyper_grad_fsla_ada(args, hyt, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num, v_state, v_momentum, v_adaptive)

                    tmp_grad = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_grad, grad)]
                    tmp_v_state = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_v_state, v_s)]
                    tmp_v_momentum = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_v_momentum, v_m)]
                    tmp_v_adaptive = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_v_adaptive, v_ada)]
                    tmp_v_norm = [tmp_g + fast_g/args.task_num for tmp_g, fast_g in zip(tmp_v_norm, v_norm)]
                    tmp_loss.append(loss)

                v_state = tmp_v_state; v_momentum = tmp_v_momentum; v_adaptive = tmp_v_adaptive

                if hyt % args.interval  == 0:
                    logger.update_err(np.mean(tmp_loss))
                    logger.update_v_norm(tmp_v_norm)


                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), tmp_grad):
                    p.grad = g.detach().clone()
                opt_lamda.step()


            elif args.alg == 'MRBO': #or SUSTAIN
                
                eta = args.d / (args.m + hyt + 1)**(1/3)
                alpha_lamda = args.c_lamda * eta ** 2
                alpha_inner = args.c_inner * eta ** 2
                # alpha_lamda = 0.9
                # alpha_inner = 0.9
                print(eta, alpha_lamda, alpha_inner)
                if hyt == 0:
                    grad_lamda = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, tmp_grad)]
                    
                    if hyt % args.interval  == 0:logger.update_err(loss)
                else:
                    grad_lamda_cur = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda_cur = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_cur, tmp_grad)]
                    
                    if hyt % args.interval  == 0:logger.update_err(loss)
                    
                    grad_lamda_old = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model_old, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model_old, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda_old = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_old, tmp_grad)]


                    grad_lamda = [g_cur + (1 - alpha_lamda) * (g - g_old) for g_cur, g_old, g in zip(grad_lamda_cur, grad_lamda_old, grad_lamda)]

                model_old = deepcopy(model)

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = g.detach().clone()
                opt_lamda.step()

            
            elif args.alg == 'MSTSA':
                eta = 1 / (hyt + 1)**(1/2)
                # print(eta)
                if hyt == 0:
                    grad_lamda = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, tmp_grad)]
                    
                    if hyt % args.interval  == 0:logger.update_err(loss)
                else:
                    grad_lamda_cur = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda_cur = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_cur, tmp_grad)]
                    
                    if hyt % args.interval  == 0:logger.update_err(loss)

                    grad_lamda_old = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model_old, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model_old, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda_old = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_old, tmp_grad)]

                    grad_lamda = [g_cur + (1 - args.c_lamda) * (g - g_old) \
                        for g_cur, g_old, g in zip(grad_lamda_cur, grad_lamda_old, grad_lamda)]

                model_old = deepcopy(model)

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = eta * g.detach().clone()
                opt_lamda.step()


            elif args.alg == 'VRBO':
                if hyt % args.spider_iters == 0:

                    x_spt_spider, y_spt_spider = spider_batch['train']
                    x_qry_spider, y_qry_spider = spider_batch['test']
                    x_spt_spider, y_spt_spider, x_qry_spider, y_qry_spider = x_spt_spider.cuda(), y_spt_spider.cuda(), x_qry_spider.cuda(), y_qry_spider.cuda()

                    grad_lamda = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt_spider, y_spt_spider, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt_spider, y_spt_spider, x_qry_spider, y_qry_spider, t_num)
                        grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, tmp_grad)]

                grad_lamda_cur = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                for t_num in range(args.task_num):  
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                    grad_lamda_cur = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_cur, tmp_grad)]
                
                if hyt % args.interval  == 0:logger.update_err(loss)
                
                grad_lamda_old = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                for t_num in range(args.task_num):  
                    new_params = inner_update(args, model_old, x_spt, y_spt, t_num)
                    loss, tmp_grad = hyper_grad_ns(args, model_old, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                    grad_lamda_old = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_old, tmp_grad)]

                grad_lamda = [g_cur +  (g - g_old) for g_cur, g_old, g in zip(grad_lamda_cur, grad_lamda_old, grad_lamda)]

                model_old = deepcopy(model)

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad =  g.detach().clone()
                opt_lamda.step()

            
            elif args.alg == 'STABLE': 
                if hyt == 0:
                    H_xy = torch.zeros([hyperDim, innerDim]).cuda()
                    H_yy = torch.zeros([innerDim, innerDim]).cuda()

                h_xy_k0 = torch.zeros([innerDim, hyperDim]).cuda()
                h_xy_k1 = torch.zeros([innerDim, hyperDim]).cuda()
                h_yy_k0  = torch.zeros([innerDim, innerDim]).cuda()
                h_yy_k1  = torch.zeros([innerDim, innerDim]).cuda()
                for t_num in range(args.task_num):
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    grad_inner = concat(grad_normal(args, model, x_spt, y_spt, t_num, create_graph=True, new_params=new_params))
                    new_params_old = inner_update(args, model_old, x_spt, y_spt, t_num)
                    grad_inner_old = concat(grad_normal(args, model_old, x_spt, y_spt, t_num, create_graph=True, new_params=new_params_old))

                    h_xy_k0_tmp, h_xy_k1_tmp, h_yy_k0_tmp, h_yy_k1_tmp = [], [], [], []

                    for index in range(grad_inner.size()[0]):
                        h_xy_k0_tmp.append(concat(torch.autograd.grad(grad_inner_old[index], model_old.getHyperRep_params(), retain_graph=True)))
                        h_xy_k1_tmp.append(concat(torch.autograd.grad(grad_inner[index], model.getHyperRep_params(), retain_graph=True)))

                        h_yy_k0_tmp.append(concat(torch.autograd.grad(grad_inner_old[index], new_params_old, retain_graph=True)))
                        h_yy_k1_tmp.append(concat(torch.autograd.grad(grad_inner[index], new_params, retain_graph=True)))

                    h_xy_k0_tmp, h_xy_k1_tmp, h_yy_k0_tmp, h_yy_k1_tmp = torch.stack(h_xy_k0_tmp), torch.stack(h_xy_k1_tmp), torch.stack(h_yy_k0_tmp),torch.stack(h_yy_k1_tmp)
                    h_xy_k0 = h_xy_k0 + h_xy_k0_tmp.detach().clone()/args.task_num 
                    h_xy_k1 = h_xy_k1 + h_xy_k1_tmp.detach().clone()/args.task_num 
                    h_yy_k0 = h_yy_k0 + h_yy_k0_tmp.detach().clone()/args.task_num
                    h_yy_k1 = h_yy_k1 + h_yy_k1_tmp.detach().clone()/args.task_num

                H_xy = (1-args.tau)*(H_xy-torch.t(h_xy_k0))+torch.t(h_xy_k1)
                H_yy = (1-args.tau)*(H_yy-torch.t(h_yy_k0))+torch.t(h_yy_k1) + torch.diag(0.01 * torch.ones(H_yy.shape[0])).cuda()
                
                grad_lamda = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                for t_num in range(args.task_num):
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=new_params)

                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

                    ld_update = concat(grad_o_ld) - torch.matmul(torch.matmul(H_xy, torch.inverse(H_yy)), concat(grad_o_w))
                    grad_lamda_tmp = split_as_model(model.getHyperRep_params(), ld_update)
                    grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, grad_lamda_tmp)]

                if hyt % args.interval  == 0:logger.update_err(dev_loss.data.cpu().numpy().item())
                model_old = deepcopy(model)

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = g.detach().clone()
                opt_lamda.step()


            elif args.alg == 'SMB':
                eta = 1 / (hyt + 1)**(1/2)

                if hyt == 0:
                    grad_lamda = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, tmp_grad)]
                    
                    if hyt % args.interval  == 0:logger.update_err(loss)
                else:
                    grad_lamda_cur = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda_cur = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_cur, tmp_grad)]
                    grad_lamda = args.c_lamda * grad_lamda_cur + (1 - args.c_lamda) * grad_lamda
                    if hyt % args.interval  == 0:logger.update_err(loss)

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = eta * g.detach().clone()
                opt_lamda.step()


            elif args.alg == 'AsBio':
                if hyt == 0: exp_avg_sq = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]

                if hyt % args.spider_iters == 0:

                    x_spt_spider, y_spt_spider = spider_batch['train']
                    x_qry_spider, y_qry_spider = spider_batch['test']
                    x_spt_spider, y_spt_spider, x_qry_spider, y_qry_spider = x_spt_spider.cuda(), y_spt_spider.cuda(), x_qry_spider.cuda(), y_qry_spider.cuda()

                    grad_lamda = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.spider_batch_size):  
                        new_params = inner_update(args, model, x_spt_spider, y_spt_spider, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt_spider, y_spt_spider, x_qry_spider, y_qry_spider, t_num)
                        grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, tmp_grad)]
                    
                    if hyt % args.interval  == 0:logger.update_err(loss)
                else:

                    grad_lamda_cur = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda_cur = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_cur, tmp_grad)]
                    
                    if hyt % args.interval  == 0:logger.update_err(loss)
                    
                    grad_lamda_old = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                    for t_num in range(args.task_num):  
                        new_params = inner_update(args, model_old, x_spt, y_spt, t_num)
                        loss, tmp_grad = hyper_grad_ns(args, model_old, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                        grad_lamda_old = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_old, tmp_grad)]
                    
                    grad_lamda = [g_cur +  args.storm_coef *(g - g_old) for g_cur, g_old, g in zip(grad_lamda_cur, grad_lamda_old, grad_lamda)]

                exp_avg_sq = [args.beta_adam * sq +  (1 - args.beta_adam) * g_ld ** 2 for sq, g_ld in zip(exp_avg_sq, grad_lamda)]

                model_old = deepcopy(model)

                opt_lamda.zero_grad()
                for p, g, sq in zip(model.getHyperRep_params(), grad_lamda, exp_avg_sq):
                    p.grad = g.detach().clone()/ (sq + eps)**0.5
                opt_lamda.step()

                if args.lasso:
                    thre = []
                    for sq in exp_avg_sq:
                        thre.append(args.hlr * args.th/ (sq + eps)**0.5)
                    newHyper = soft_th(model.getHyperRep_params(), threshold=thre)
                    model.setHyperRep_params(newHyper)


            elif args.alg == 'VR-BiAdam':
                if hyt == 0: 
                    exp_avg_sq = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]

                grad_lamda_cur = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                for t_num in range(args.task_num):  
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                    grad_lamda_cur = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_cur, tmp_grad)]

                if hyt % args.interval  == 0:logger.update_err(loss)

                grad_lamda_old = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                for t_num in range(args.task_num):  
                    new_params = inner_update(args, model_old, x_spt, y_spt, t_num)
                    loss, tmp_grad = hyper_grad_ns(args, model_old, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                    grad_lamda_old = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_old, tmp_grad)]
                
                if hyt == 0:
                    grad_lamda = grad_lamda_cur
                else:
                    grad_lamda = [g_cur +  args.storm_coef *(g - g_old) for g_cur, g_old, g in zip(grad_lamda_cur, grad_lamda_old, grad_lamda)]

                exp_avg_sq = [args.beta_adam * sq +  (1 - args.beta_adam) * g_ld ** 2 for sq, g_ld in zip(exp_avg_sq, grad_lamda)]

                model_old = deepcopy(model)

                opt_lamda.zero_grad()
                for p, g, sq in zip(model.getHyperRep_params(), grad_lamda, exp_avg_sq):
                    p.grad = g.detach().clone()/ (sq + eps)**0.5
                opt_lamda.step()


            elif args.alg == 'BiAdam':
                if hyt == 0: 
                    exp_avg_sq = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]

                grad_lamda_cur = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                for t_num in range(args.task_num):  
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    loss, tmp_grad = hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num)
                    grad_lamda_cur = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda_cur, tmp_grad)]
                
                if hyt % args.interval  == 0:logger.update_err(loss)
                
                if hyt == 0:
                    grad_lamda = grad_lamda_cur
                else:
                    grad_lamda = [(1 - args.storm_coef) * g_cur +  args.storm_coef * g for g_cur, g in zip(grad_lamda_cur, grad_lamda)]

                exp_avg_sq = [args.beta_adam * sq +  (1 - args.beta_adam) * g_ld ** 2 for sq, g_ld in zip(exp_avg_sq, grad_lamda)]

                model_old = deepcopy(model)

                opt_lamda.zero_grad()
                for p, g, sq in zip(model.getHyperRep_params(), grad_lamda, exp_avg_sq):
                    p.grad = g.detach().clone()/ (sq + eps)**0.5
                opt_lamda.step()

        
            elif args.alg == 'HFBiO_vanilla':
                if hyt == 0:
                    Jacobian = torch.zeros([hyperDim, innerDim]).cuda()

                h_xy = torch.zeros([innerDim, hyperDim]).cuda()
                h_yy  = torch.zeros([innerDim, innerDim]).cuda()

                grad_lamda = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                for t_num in range(args.task_num):
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    grad_inner = concat(grad_normal(args, model, x_spt, y_spt, t_num, create_graph=True, new_params=new_params))

                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o = torch.autograd.grad(outputs=dev_loss, inputs= list(model.getHyperRep_params())+new_params)
                    grad_o_w, grad_o_ld = grad_o[16:], grad_o[:16]

                    ld_update = concat(grad_o_ld) + torch.matmul(Jacobian, concat(grad_o_w))
                    grad_lamda_tmp = split_as_model(model.getHyperRep_params(), ld_update)
                    grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, grad_lamda_tmp)]

                    h_xy_tmp, h_yy_tmp = [], []

                    for index in range(grad_inner.size()[0]):
                        h_xy_tmp.append(concat(torch.autograd.grad(grad_inner[index], model.getHyperRep_params(), retain_graph=True)))
                        h_yy_tmp.append(concat(torch.autograd.grad(grad_inner[index], new_params, retain_graph=True)))

                    h_xy_tmp, h_yy_tmp = torch.stack(h_xy_tmp), torch.stack(h_yy_tmp)
                    h_xy = h_xy + h_xy_tmp.detach().clone()/args.task_num 
                    h_yy = h_yy + h_yy_tmp.detach().clone()/args.task_num 

                Jacobian -= args.tau * torch.t(h_xy + torch.matmul(h_yy, torch.t(Jacobian)))
                if hyt % args.interval  == 0:logger.update_err(dev_loss.data.cpu().numpy().item())

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = g.detach().clone()
                opt_lamda.step()

            
            elif args.alg == 'HFBiO':
                if hyt == 0:
                    Jacobian = torch.zeros([hyperDim, innerDim]).cuda()

                h_xy = torch.zeros([innerDim, hyperDim]).cuda()
                h_yy  = torch.zeros([innerDim, innerDim]).cuda()
                for t_num in range(args.task_num):
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    grad_inner = concat(grad_normal(args, model, x_spt, y_spt, t_num, new_params=new_params))

                    hyper_noise = []; hyper_dir = []; inner_noise = []; inner_dir = []
                    for q in range(args.Q):
                        noise_hyper = [torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in model.getHyperRep_params()]
                        hyper_params = [p + args.mu * n for p, n in zip(model.getHyperRep_params(), noise_hyper)]
                        hyper_params.extend(new_params)
                        grad_inner_hyper = concat(grad_normal(args, model, x_spt, y_spt, t_num, new_params=hyper_params))
                        grad_hyper_dir = (grad_inner_hyper[-innerDim:] - grad_inner) / args.mu

                        hyper_noise.append(concat(noise_hyper))
                        hyper_dir.append(grad_hyper_dir)

                        noise_inner = [torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in model.getInner_params()]
                        inner_params = [p + args.niu * n for p, n in zip(new_params, noise_inner)]
                        hyper_params = list(model.getHyperRep_params())
                        hyper_params.extend(inner_params)
                        grad_inner_inner = concat(grad_normal(args, model, x_spt, y_spt, t_num, new_params=hyper_params))
                        grad_inner_dir = (grad_inner_inner[-innerDim:] - grad_inner) / args.niu

                        inner_noise.append(concat(noise_inner))
                        inner_dir.append(grad_inner_dir)
                    
                    hyper_noise = torch.stack(hyper_noise)
                    hyper_dir = torch.stack(hyper_dir)

                    inner_noise = torch.stack(inner_noise)
                    inner_dir = torch.stack(inner_dir)

                    h_xy += torch.matmul(torch.t(hyper_dir), hyper_noise)/args.Q/args.task_num
                    h_yy += torch.matmul(torch.t(inner_dir), inner_noise)/args.Q/args.task_num

                
                Jacobian -= args.tau * torch.t(h_xy + torch.matmul(h_yy, torch.t(Jacobian)))

                grad_lamda = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                for t_num in range(args.task_num):
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=new_params)

                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

                    ld_update = concat(grad_o_ld) + torch.matmul(Jacobian, concat(grad_o_w))
                    grad_lamda_tmp = split_as_model(model.getHyperRep_params(), ld_update)
                    grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, grad_lamda_tmp)]

                if hyt % args.interval  == 0:logger.update_err(dev_loss.data.cpu().numpy().item())

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = g.detach().clone()
                opt_lamda.step()
  

            elif args.alg == 'HFBiO_special':
                if hyt == 0:
                    Jacobian = torch.zeros([hyperDim, innerDim]).cuda()
                    noise_hyper_all = [[[[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in model.getHyperRep_params()] \
                        for _ in range(args.Q)] for _ in range(args.task_num)] for _ in range(args.T)]
                    noise_inner_all = [[[[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in model.getInner_params()] \
                        for _ in range(args.Q)] for _ in range(args.task_num)] for _ in range(args.T)]
                
                grad_lamda = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                hyper_noise = []; hyper_dir = []; inner_noise = []; inner_dir = []

                for t_num in range(args.task_num):

                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    grad_inner = concat(grad_normal(args, model, x_spt, y_spt, t_num, new_params=new_params))
                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o = torch.autograd.grad(outputs=dev_loss, inputs= list(model.getHyperRep_params())+new_params)
                    grad_o_w, grad_o_ld = grad_o[16:], grad_o[:16]
                    ld_update = concat(grad_o_ld) + torch.matmul(Jacobian, concat(grad_o_w))
                    grad_lamda_tmp = split_as_model(model.getHyperRep_params(), ld_update)
                    grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(grad_lamda, grad_lamda_tmp)]

                    for q in range(args.Q):
                        noise_hyper = noise_hyper_all[hyt][t_num][q]; noise_inner = noise_inner_all[hyt][t_num][q]
                        hyper_params = [p + args.mu * n for p, n in zip(model.getHyperRep_params(), noise_hyper)]

                        hyper_params.extend(new_params)
    
                        grad_inner_hyper = concat(grad_normal(args, model, x_spt, y_spt, t_num, new_params=hyper_params))
                        grad_hyper_dir = (grad_inner_hyper[-innerDim:] - grad_inner) / args.mu

                        hyper_noise.append(concat(noise_hyper))
                        hyper_dir.append(grad_hyper_dir)

                        noise_inner = [torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in model.getInner_params()]
                        inner_params = [p + args.niu * n for p, n in zip(new_params, noise_inner)]
                        hyper_params = list(model.getHyperRep_params())
                        hyper_params.extend(inner_params)
                        grad_inner_inner = concat(grad_normal(args, model, x_spt, y_spt, t_num, new_params=hyper_params))
                        grad_inner_dir = (grad_inner_inner[-innerDim:] - grad_inner) / args.niu

                        inner_noise.append(concat(noise_inner))
                        inner_dir.append(grad_inner_dir)

                if hyt % args.interval  == 0: logger.update_err(dev_loss.data.cpu().numpy().item())


                hyper_noise = torch.stack(hyper_noise)
                hyper_dir = torch.stack(hyper_dir)
                inner_noise = torch.stack(inner_noise)
                inner_dir = torch.stack(inner_dir)
                h_xy = torch.matmul(torch.t(hyper_dir), hyper_noise)/args.task_num
                Jacobian -= args.tau * torch.t(h_xy + torch.matmul(torch.t(inner_dir), torch.matmul(inner_noise, torch.t(Jacobian)))/args.task_num)

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = g.detach().clone()
                opt_lamda.step()
             

            elif args.alg == 'ESJ':
                if hyt == 0:
                    noise_hyper_all = [[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in model.getHyperRep_params()] \
                        for _ in range(20000)]
                grad_lamda = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                for t_num in range(args.task_num):
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=new_params)

                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
                    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

                    for q in range(args.Q):
                        noise_hyper = noise_hyper_all[np.random.randint(20000)]
                        hyper_params = [p + args.mu * n for p, n in zip(model.getHyperRep_params(), noise_hyper)]
                        new_params_hyper= inner_update(args, model, x_spt, y_spt, t_num, hyper_params=hyper_params)
                        hyper_dir = (concat(new_params_hyper) - concat(new_params)) / args.mu

                        hyper_noise = concat(noise_hyper).view(1,-1)
                        hyper_dir = hyper_dir.view(1,-1)

                        ld_update = concat(grad_o_ld) + (torch.matmul(hyper_dir, concat(grad_o_w).view(-1,1)).item() * hyper_noise).view(-1)
                        grad_lamda_tmp = split_as_model(model.getHyperRep_params(), ld_update)
                        grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num/args.Q for tmp_g, fast_g in zip(grad_lamda, grad_lamda_tmp)]

                if hyt % args.interval  == 0: logger.update_err(dev_loss.data.cpu().numpy().item())

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = g.detach().clone()
                opt_lamda.step()

            
            elif args.alg == 'HOZOJ':
                if hyt == 0:
                    noise_hyper_all = [[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in model.getHyperRep_params()] \
                        for _ in range(20000)]
                grad_lamda = [torch.zeros_like(p).cuda() for p in model_old.getHyperRep_params()]
                for t_num in range(args.task_num):
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    logits_q = model(x_qry[t_num], new_params, bn_training=True)
                    dev_loss = F.cross_entropy(logits_q, y_qry[t_num]).item()

                    for q in range(args.Q):
                        noise_hyper = noise_hyper_all[np.random.randint(20000)]
                        hyper_params = [p + args.mu * n for p, n in zip(model.getHyperRep_params(), noise_hyper)]
                        new_params_hyper= inner_update(args, model, x_spt, y_spt, t_num, hyper_params=hyper_params)

                        logits_q_hyper = model(x_qry[t_num], new_params_hyper, bn_training=True)
                        dev_loss_hyper = F.cross_entropy(logits_q_hyper, y_qry[t_num]).item()

                        ld_update = (dev_loss_hyper - dev_loss) * concat(noise_hyper) /args.mu
                        grad_lamda_tmp = split_as_model(model.getHyperRep_params(), ld_update)
                        grad_lamda = [tmp_g + fast_g.detach().clone()/args.task_num/args.Q for tmp_g, fast_g in zip(grad_lamda, grad_lamda_tmp)]

                if hyt % args.interval  == 0: logger.update_err(dev_loss)
                print(dev_loss)

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), grad_lamda):
                    p.grad = g.detach().clone()
                opt_lamda.step()          


            ######################################################################################################
            training_time = time.time() - start_time
            
            if hyt % args.interval == 0:
                logger.update_time(training_time)


                accs = []; losses = []
                test_step = 0                
                model.eval()
                for test_batch in data_loader.dataloader_val:
                    x_spt, y_spt = test_batch['train']
                    x_qry, y_qry = test_batch['test']
                    x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        test_acc, test_loss = finetunning(args, model, x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        accs.append(test_acc); losses.append(test_loss)
                    
                    test_step += args.task_num
                    if test_step > 100:
                        break

                model.train()
                logger.update_testAcc(np.mean(accs))
                logger.print(hyt)
                logger.save()


            hyt += 1
            if hyt >= args.T:
                return

if __name__ == "__main__":
    # sending arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='AsBio', choices=['stocBiO', 'HOAG', 'TTSA', 'BSA',  'ESJ', 'HOZOJ', 'BiAdam', 'VR-BiAdam',
                                                        'reverse', 'AID_CG', 'AID_NS', 'VRBO', 'MRBO', 'MSTSA', 'Dire',\
                                                            'STABLE', 'AsBio', 'FSLA', 'FSLA_ADA', 'SMB', 'SVRB', 'HFBiO_vanilla', 'HFBiO', 'HFBiO_special'])
    parser.add_argument('--data', type=str, default='Omniglot', choices=['Omniglot', 'MiniImageNet'])                    
    parser.add_argument('--outer_opt', type=str, default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--innerT', type=int, default= 4, help="Number of Inner Iters")
    parser.add_argument('--T', type=int, default=2000, help="Number of Outer Iters")
    parser.add_argument('--n_way', type=int, help='number classes', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='number samples for query set', default=15)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    parser.add_argument('--imgc', type=int, help='imgc', default=1)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--v_iter', type=int, default=3, help="Number of iterations to compute v")
    parser.add_argument('--spider_iters', type=int, default=3, help="Spider Frequency")
    parser.add_argument('--spider_batch_size', type=int, default=16, help="Spider Batch_size")
    parser.add_argument('--hlr', type=float, default=0.1, help="HyperLr")
    parser.add_argument('--lr', type=float, default= 0.4, help="InnerLr")
    parser.add_argument('--beta', type=float, default= 0.5, help="Shrinkage parameter used in Neumann series")
    parser.add_argument('--storm_coef', type=float, default= 1, help="")
    parser.add_argument('--beta_adam', type=float, default= 0.99, help="Exponetial Moving Average Coefficient")
    parser.add_argument('--lamda', type=float, default= 0.01, help="Regularization parameter for v using conjugate")
    parser.add_argument('--interval', type=int, help='', default=1)
    
    ##hyper-params for MRBO
    parser.add_argument('--d', type=float, default=10, help="")
    parser.add_argument('--m', type=float, default=500, help="")
    parser.add_argument('--c_lamda', type=float, default=0.9, help="")
    parser.add_argument('--c_inner', type=float, default=0.9, help="")

    #hyper-params for STABLE/HFBiO
    parser.add_argument('--tau', type=float, default=0.5, help="")
    parser.add_argument('--mu', type=float, default=1, help="")
    parser.add_argument('--niu', type=float, default=1, help="")
    parser.add_argument('--Q', type=int, default=1, help="")

    parser.add_argument('--v_beta1', type=float, default= 0.9, help="Exponetial Moving Average Coefficient")
    parser.add_argument('--v_beta2', type=float, default= 0.99, help="Exponetial Moving Average Coefficient")

    args = parser.parse_args()
    config = args

    prefix_dir =  '/ocean/projects/cis220038p/junyili/AdaBilevel'

    if args.data == 'Omniglot':
        data_loader = OmniglotNShot(batchsz=args.task_num, n_way = args.n_way, k_shot=args.k_spt, k_query=args.k_qry, spider_batchsz=args.spider_batch_size)
    elif args.data == 'MiniImageNet':
        data_loader = MiniImagenetNShot(batchsz=args.task_num, n_way = args.n_way, k_shot=args.k_spt, k_query=args.k_qry)

    prefix = str(args.n_way) + '-' + str(args.k_spt) + '-' + str(args.k_qry) + '-' + str(args.task_num) + '-' + args.alg + '-' + args.outer_opt

    postfix = 'T-' + str(args.T) + '-hlr-' + str(args.hlr) + '-innerT-' + str(args.innerT) + '-lr-' + str(args.lr)
    if args.alg == 'AID_CG':
        postfix += '-v_iter-' + str(args.v_iter) + '-lamda-' + str(args.lamda) + '-beta-' + str(args.beta)
    elif args.alg == 'AID_NS':
        postfix += '-v_iter-' + str(args.v_iter) + '-beta-' + str(args.beta)
    
    elif args.alg == 'FSLA':
        postfix +=  '-beta-' + str(args.beta) + '-v_iter-' + str(args.v_iter)

    elif args.alg == 'FSLA_ADA':
        postfix += '-beta-' + str(args.beta) + '-beta1-' + str(args.v_beta1) + '-beta2-' + str(args.v_beta2) + '-v_iter-' + str(args.v_iter)

    elif args.alg == 'VRBO':
        postfix += '-spider_iters-' + str(args.spider_iters) + '-spider_bs-' + str(args.spider_batch_size)
    
    elif args.alg == 'AsBio':
        postfix += '-spider_iters-' + str(args.spider_iters) + '-spider_bs-' + str(args.spider_batch_size) + '-beta_adam-' + str(args.beta_adam) + '-s_coef-' + str(args.storm_coef)
    
    elif args.alg == 'BiAdam':
        postfix +=  '-beta_adam-' + str(args.beta_adam) + '-s_coef-' + str(args.storm_coef)
    
    elif args.alg == 'VR-BiAdam':
        postfix +=  '-beta_adam-' + str(args.beta_adam) + '-s_coef-' + str(args.storm_coef)
    
    elif args.alg == 'MRBO':
        postfix += '-d-' + str(args.d) + '-m-' + str(args.m) + '-c_lamda-' + str(args.c_lamda) + '-c_inner-' + str(args.c_inner)
    
    elif args.alg == 'MSTSA':
        postfix +=  '-c_lamda-' + str(args.c_lamda)
    
    elif args.alg == 'STABLE':
        postfix += '-tau-' + str(args.tau)
    
    elif args.alg == 'SMB':
        postfix += '-c_lamda-' + str(args.c_lamda) + '-c_inner-' + str(args.c_inner)
    
    elif args.alg == 'HFBiO_vanilla':
        postfix += '-tau-' + str(args.tau) + '-' + args.cr
    
    elif args.alg == 'HFBiO':
        postfix += '-tau-' + str(args.tau) + '-mu-' + str(args.mu) + '-niu-' + str(args.niu) + '-Q-' + str(args.Q) + '-' + args.cr
    
    elif args.alg == 'HFBiO_special':
        postfix += '-tau-' + str(args.tau) + '-mu-' + str(args.mu) + '-niu-' + str(args.niu) + '-Q-' + str(args.Q) + '-' + args.cr
    
    elif args.alg == 'ESJ':
        postfix +=  '-mu-' + str(args.mu) + '-Q-' + str(args.Q) + '-' + args.cr
    
    elif args.alg == 'HOZOJ':
        postfix +=  '-mu-' + str(args.mu) + '-Q-' + str(args.Q) + '-' + args.cr
    
    logger = Logger_meta(prefix_dir + '/hyper_rep', prefix = prefix, postfix= postfix)

    train(config, data_loader, logger)
