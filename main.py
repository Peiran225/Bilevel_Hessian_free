from multiprocessing import reduction
import os
import time
import pdb
import logging
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from utils import *

import pdb

def train(args, data_loader, logger):     

    eps = 1e-6

    #inner variable
    # config = [
    #     ('linear', [args.num_class, 28*28]),
    # ]

    # config = [
    #     ('linear', [16*16, 28*28]),
    #     ('relu', [True]),        
    #     ('linear', [8*8, 16*16]),
    #     ('relu', [True]),
    #     ('linear', [args.num_class, 8*8]),
    # ]
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
        ('linear', [10, 64])
    ]


    inner = Learner(config).cuda()
    inner_old = Learner(config).cuda()

    for p, p_old in zip(inner.parameters(), inner_old.parameters()):
        p_old.data = p.data
    # pdb.set_trace()
    ld = torch.tensor(np.random.normal(size=args.num_sample), dtype=torch.float).cuda().requires_grad_()
    ld_old = torch.zeros_like(ld).requires_grad_()
    ld_old.data = ld.data

    if args.cr == 'CE':
        args.crit = nn.CrossEntropyLoss(reduction='none')
        args.crit_mean = nn.CrossEntropyLoss()
    elif args.cr == 'SH':
        args.crit = nn.MultiMarginLoss(p=2, reduction='none')
        args.crit_mean = nn.MultiMarginLoss(p=2)


    if args.outer_opt == 'SGD':
        opt_lamda = optim.SGD([ld,], lr=args.hlr)
    elif args.outer_opt == 'Adam':
        opt_lamda = optim.Adam([ld,], lr=args.hlr)

    ###########################Main training loop#################################################
    for hyt in range(args.T):
        start_time = time.time()
        if args.alg == 'reverse':
            new_params = [p for p in inner.parameters()]
            for _ in range(args.innerT):
                cnt, x, y = data_loader.get_batch_train(args.batch_size)
                loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(ld[cnt]))
                grad = torch.autograd.grad(outputs=loss, inputs= inner.parameters(), create_graph=True)
                new_params = [p - args.lr * g for p, g in zip(new_params,grad)]

            devx, devy = data_loader.get_val()
            loss = args.crit_mean(inner(devx, new_params), devy); logger.update_err(loss.data.cpu().numpy().item())
            grad_lamda = torch.autograd.grad(outputs=loss, inputs=[ld])[0].detach().clone() 

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            if args.lasso:
                print('check check!')
                ld = soft_th(ld, threshold=args.th)

            for p, new_p in zip(inner.parameters(), new_params):
                p.data = new_p.data

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 


        elif args.alg == 'AID_CG':                             
            inner = inner_update(args, args.batch_size, inner, ld, data_loader)
            loss, grad_lamda = hyper_grad_cg(args, args.batch_size, inner, ld, data_loader)
            logger.update_err(loss)

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 


        elif args.alg == 'AID_NS':  # or stocBio if without warm start/ or AID_FP / or BSA
            inner = inner_update(args, args.batch_size, inner, ld, data_loader)
            loss, grad_lamda = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)
            logger.update_err(loss)

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 


        elif args.alg == 'FSLA': 
            if hyt == 0:
                v_state = []
                for p in inner.parameters():
                    v_state.append(torch.zeros_like(p).cuda())

            inner = inner_update(args, args.batch_size, inner, ld, data_loader)
            loss, grad_lamda, v_state, v_norm = hyper_grad_fsla(args, args.batch_size, inner, ld, data_loader, v_state)
            logger.update_err(loss)

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 
            logger.update_v_norm(v_norm)

        elif args.alg == 'FSLA_ADA': 
            if hyt == 0:
                v_state = []; v_momentum = []; v_adaptive = []
                for p in inner.parameters():
                    v_state.append(torch.zeros_like(p).cuda())
                    v_momentum.append(torch.zeros_like(p).cuda())
                    v_adaptive.append(torch.zeros_like(p).cuda())

            inner = inner_update(args, args.batch_size, inner, ld, data_loader)
            loss, grad_lamda, v_state, v_momentum, v_adaptive, v_norm = \
                hyper_grad_fsla_ada(args, hyt, args.batch_size, inner, ld, data_loader, v_state, v_momentum, v_adaptive)
            logger.update_err(loss)

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 
            logger.update_v_norm(v_norm)

            if hyt % 10 == 0:
                print(v_norm)


        elif args.alg == 'MRBO':
            eta = args.d / (args.m + hyt + 1)**(1/3)
            alpha_lamda = args.c_lamda * eta ** 2
            alpha_inner = args.c_inner * eta ** 2
            print(eta, alpha_lamda, alpha_inner)
            if hyt == 0:
                loss, grad_lamda = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)
                grad_inner = grad_normal(args, args.batch_size, inner, ld, data_loader)
            else:
                cnt, x, y = data_loader.get_batch_train(args.batch_size)
                loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(ld[cnt]))
                grad_inner_cur = torch.autograd.grad(outputs=loss, inputs= inner.parameters())

                loss_old = torch.mean(args.crit(inner_old(x), y)*torch.sigmoid(ld_old[cnt]))
                grad_inner_old = torch.autograd.grad(outputs=loss_old, inputs= inner_old.parameters())
                _, grad_lamda_cur, grad_lamda_old = hyper_grad_ns_double(args, args.batch_size, inner, ld, inner_old, ld_old, data_loader)

                grad_lamda = grad_lamda_cur + (1 - alpha_lamda) * (grad_lamda - grad_lamda_old)
                grad_inner = [g_cur + (1 - alpha_inner) * (g - g_old) \
                    for g_cur, g_old, g in zip(grad_inner_cur, grad_inner_old, grad_inner)]

            ld_old.data = ld.data

            opt_lamda.zero_grad()
            ld.grad = eta * grad_lamda.detach().clone()
            opt_lamda.step()
            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())

            for p, p_old in zip(inner.parameters(), inner_old.parameters()):
                p_old.data = p.data
            
            for p, g in zip(inner.parameters(),grad_inner):
                p.detach_()
                p -=  args.lr * eta * g
                p.requires_grad_()

            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'MSTSA':
            eta = 1 / (hyt + 1)**(1/2)
            # print(eta)
            if hyt == 0:
                loss, grad_lamda = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)
                grad_inner = grad_normal(args, args.batch_size, inner, ld, data_loader)
            else:
                cnt, x, y = data_loader.get_batch_train(args.batch_size)
                loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(ld[cnt]))
                grad_inner = torch.autograd.grad(outputs=loss, inputs= inner.parameters())
                _, grad_lamda_cur, grad_lamda_old = hyper_grad_ns_double(args, args.batch_size, inner, ld, inner_old, ld_old, data_loader)


                grad_lamda = grad_lamda_cur + (1 - args.c_lamda) * (grad_lamda - grad_lamda_old)

            ld_old.data = ld.data

            opt_lamda.zero_grad()
            ld.grad = eta * grad_lamda.detach().clone()
            opt_lamda.step()
            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())

            for p, p_old in zip(inner.parameters(), inner_old.parameters()):
                p_old.data = p.data
            
            for p, g in zip(inner.parameters(),grad_inner):
                p.detach_()
                p -=  args.lr * eta * g
                p.requires_grad_()

            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'VRBO':
            if hyt % args.spider_iters == 0:
                loss, grad_lamda = hyper_grad_ns(args, args.spider_batch_size, inner, ld, data_loader)
                grad_inner = grad_normal(args, args.spider_batch_size, inner, ld, data_loader)
            # pdb.set_trace()

            ld_old.data = ld.data

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 

            inner, inner_old, grad_inner, grad_lamda = \
                inner_update_spider(args, args.batch_size, inner, inner_old, ld, ld_old, data_loader, grad_inner, grad_lamda)
        
            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'STABLE': #Temp correction hard to tune
            if hyt == 0:
                H_xy = torch.zeros([5000, 7850]).cuda()
                H_yy = torch.zeros([7850, 7850]).cuda()

            inner, inner_old, ld, ld_old, H_xy, H_yy = stable(args, args.batch_size, inner, inner_old, ld, ld_old, H_xy, H_yy, data_loader, opt_lamda)

            logger.update_gnorm(torch.norm((ld - ld_old)/args.hlr).data.cpu().numpy().item())
            # pdb.set_trace()
            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'SMB': #Double Momentum
            eta = 1 / (hyt + 1)**(1/2)

            print(eta)
            if hyt == 0:
                loss, grad_lamda = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)
                grad_inner = grad_normal(args, args.batch_size, inner, ld, data_loader)
            else:
                grad_inner_cur = grad_normal(args, args.batch_size, inner, ld, data_loader)
                grad_inner = [args.c_inner * g_cur + (1 - args.c_inner) * g \
                    for g_cur, g in zip(grad_inner_cur, grad_inner)]

                loss, grad_lamda_cur = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)
                grad_lamda = args.c_lamda * grad_lamda_cur + (1 - args.c_lamda) * grad_lamda

            ld_old.data = ld.data

            opt_lamda.zero_grad()
            ld.grad = eta * grad_lamda.detach().clone()
            opt_lamda.step()
            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())

            for p, p_old in zip(inner.parameters(), inner_old.parameters()):
                p_old.data = p.data
            
            for p, g in zip(inner.parameters(),grad_inner):
                p.detach_()
                p -=  args.lr * eta * g
                p.requires_grad_()

            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'AsBio':
            if hyt == 0: exp_avg_sq = torch.zeros_like(ld).cuda()

            if hyt % args.spider_iters == 0:
                loss, grad_lamda = hyper_grad_ns(args, args.spider_batch_size, inner, ld, data_loader)
                grad_inner = grad_normal(args, args.spider_batch_size, inner, ld, data_loader)

            ld_old.data = ld.data

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()/ (exp_avg_sq + eps)**0.5
            opt_lamda.step()

            # print('before', torch.sum(ld), ld[:10])
            if args.lasso:
                ld = soft_th(ld, threshold=args.th/(exp_avg_sq + eps)**0.5)
                print('after', torch.sum(ld), ld[:10], (args.th/(exp_avg_sq + eps)**0.5)[:10])

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 

            inner, inner_old, grad_inner, grad_lamda, exp_avg_sq = \
                inner_update_asbio(args, args.batch_size, inner, inner_old, ld, ld_old, data_loader, grad_inner, grad_lamda, exp_avg_sq)
        
            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'VR-BiAdam':
            if hyt == 0:
                loss, grad_lamda = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)
                grad_inner = grad_normal(args, args.batch_size, inner, ld, data_loader)
                
                exp_avg_sq_ld = torch.zeros_like(ld).cuda()
                exp_avg_sq_inner = [torch.zeros_like(p).cuda() for p in inner.parameters()]
            else:
                cnt, x, y = data_loader.get_batch_train(args.batch_size)
                loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(ld[cnt]))
                grad_inner_cur = torch.autograd.grad(outputs=loss, inputs= inner.parameters())

                loss_old = torch.mean(args.crit(inner_old(x), y)*torch.sigmoid(ld_old[cnt]))
                grad_inner_old = torch.autograd.grad(outputs=loss_old, inputs= inner_old.parameters())
                _, grad_lamda_cur, grad_lamda_old = hyper_grad_ns_double(args, args.batch_size, inner, ld, inner_old, ld_old, data_loader)

                grad_lamda = grad_lamda_cur + args.storm_coef * (grad_lamda - grad_lamda_old)
                grad_inner = [g_cur + args.storm_coef * (g - g_old) for g_cur, g_old, g in zip(grad_inner_cur, grad_inner_old, grad_inner)]


            exp_avg_sq_inner = [args.beta_adam * sq +  (1 - args.beta_adam) * g_ld ** 2 for sq, g_ld in zip(exp_avg_sq_inner, grad_inner)]
            exp_avg_sq_ld = args.beta_adam * exp_avg_sq_ld + (1 - args.beta_adam) * grad_lamda ** 2

            ld_old.data = ld.data
            opt_lamda.zero_grad()
            ld.grad =  grad_lamda.detach().clone()/(exp_avg_sq_ld + eps)**0.5
            opt_lamda.step()
            # print('hhhh:', torch.norm(exp_avg_sq_ld), torch.norm(grad_lamda), torch.norm(ld.grad))
            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())

            for p, p_old in zip(inner.parameters(), inner_old.parameters()):
                p_old.data = p.data
            
            for p, g, sq in zip(inner.parameters(),grad_inner, exp_avg_sq_inner):
                p.detach_()
                p -=  args.lr * g/(sq + eps)**0.5
                p.requires_grad_()

            # for p, g in zip(inner.parameters(),grad_inner):
            #     p.detach_()
            #     p -=  args.lr * g
            #     p.requires_grad_()

            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'BiAdam':
            if hyt == 0:
                loss, grad_lamda = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)
                grad_inner = grad_normal(args, args.batch_size, inner, ld, data_loader)
                
                exp_avg_sq_ld = torch.zeros_like(ld).cuda()
                exp_avg_sq_inner = [torch.zeros_like(p).cuda() for p in inner.parameters()]
            else:
                cnt, x, y = data_loader.get_batch_train(args.batch_size)
                loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(ld[cnt]))
                grad_inner_cur = torch.autograd.grad(outputs=loss, inputs= inner.parameters())
                _, grad_lamda_cur = hyper_grad_ns(args, args.batch_size, inner, ld, data_loader)

                grad_lamda = (1 - args.storm_coef) * grad_lamda_cur + args.storm_coef * grad_lamda
                grad_inner = [(1 - args.storm_coef) * g_cur + args.storm_coef * g for g_cur, g in zip(grad_inner_cur, grad_inner)]


            exp_avg_sq_inner = [args.beta_adam * sq +  (1 - args.beta_adam) * g_ld ** 2 for sq, g_ld in zip(exp_avg_sq_inner, grad_inner)]
            exp_avg_sq_ld = args.beta_adam * exp_avg_sq_ld + (1 - args.beta_adam) * grad_lamda ** 2

            ld_old.data = ld.data

            opt_lamda.zero_grad()
            ld.grad =  grad_lamda.detach().clone()/(exp_avg_sq_ld + eps)**0.5
            opt_lamda.step()
            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())

            for p, p_old in zip(inner.parameters(), inner_old.parameters()):
                p_old.data = p.data
            
            for p, g, sq in zip(inner.parameters(),grad_inner, exp_avg_sq_inner):
                p.detach_()
                p -=  args.lr * g/(sq + eps)**0.5
                p.requires_grad_()
            # for p, g in zip(inner.parameters(),grad_inner):
            #     p.detach_()
            #     p -=  args.lr * g
            #     p.requires_grad_()

            with torch.no_grad():
                cnt, devx, devy = data_loader.get_batch_val(args.batch_size)
                dev_loss = args.crit_mean(inner(devx), devy)
                logger.update_err(dev_loss.data.cpu().numpy().item())


        elif args.alg == 'HFBiO_vanilla':
            if hyt == 0:
                Jacobian = torch.zeros([5000, 7850]).cuda()

            h_xy = torch.zeros([7850, 5000]).cuda()
            h_yy  = torch.zeros([7850, 7850]).cuda()

            inner = inner_update(args, args.batch_size, inner, ld, data_loader)
            grad_inner = concat(grad_normal(args, args.batch_size, inner, ld, data_loader, create_graph=True))

            h_xy_tmp, h_yy_tmp = [], []

            for index in range(grad_inner.size()[0]):
                h_xy_tmp.append(concat(torch.autograd.grad(grad_inner[index], [ld,], retain_graph=True)))
                h_yy_tmp.append(concat(torch.autograd.grad(grad_inner[index], inner.parameters(), retain_graph=True)))

            h_xy_tmp, h_yy_tmp = torch.stack(h_xy_tmp), torch.stack(h_yy_tmp)
            h_xy = h_xy + h_xy_tmp.detach().clone()
            h_yy = h_yy + h_yy_tmp.detach().clone()
            
            Jacobian -= args.tau * torch.t(h_xy + torch.matmul(h_yy, torch.t(Jacobian)))


            _, devx, devy = data_loader.get_batch_val(args.batch_size)
            dev_loss = args.crit_mean(inner(devx), devy)
            grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=inner.parameters())

            grad_lamda = torch.matmul(Jacobian, concat(grad_o_w))

            logger.update_err(dev_loss.data.cpu().numpy().item())
            # pdb.set_trace()
            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 


        elif args.alg == 'HFBiO':
            if hyt == 0:
                Jacobian = torch.zeros([5000, 7850]).cuda()

            h_xy = torch.zeros([7850, 5000]).cuda()
            h_yy  = torch.zeros([7850, 7850]).cuda()

            inner = inner_update(args, args.batch_size, inner, ld, data_loader)

            hyper_noise = []; hyper_dir = []; inner_noise = []; inner_dir = []
            for q in range(args.Q):
                cnt, x, y = data_loader.get_batch_train(args.batch_size)
                loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(ld[cnt]))
                grad_inner = concat(torch.autograd.grad(outputs=loss, inputs= inner.parameters()))

                noise_hyper = [torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in [ld,]]
                hyper_params = [p + args.mu * n for p, n in zip([ld,], noise_hyper)]
                loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(hyper_params[0][cnt]))
                grad_inner_hyper = concat(torch.autograd.grad(outputs=loss, inputs= inner.parameters()))
                grad_hyper_dir = (grad_inner_hyper - grad_inner) / args.mu

                hyper_noise.append(concat(noise_hyper))
                hyper_dir.append(grad_hyper_dir)

                noise_inner = [torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in inner.parameters()]
                inner_params = [p + args.niu * n for p, n in zip(inner.parameters(), noise_inner)]
                inner_noise_model = Learner(config).cuda()
                for p, p_noise in zip(inner_params, inner_noise_model.parameters()):
                    p_noise.data = p.data
                loss = torch.mean(args.crit(inner_noise_model(x), y)*torch.sigmoid(ld[cnt]))
                grad_inner_inner = concat(torch.autograd.grad(outputs=loss, inputs= inner_noise_model.parameters()))
                grad_inner_dir = (grad_inner_inner - grad_inner) / args.niu

                inner_noise.append(concat(noise_inner))
                inner_dir.append(grad_inner_dir)
            
            hyper_noise = torch.stack(hyper_noise)
            hyper_dir = torch.stack(hyper_dir)

            inner_noise = torch.stack(inner_noise)
            inner_dir = torch.stack(inner_dir)

            h_xy += torch.matmul(torch.t(hyper_dir), hyper_noise)/args.Q
            h_yy += torch.matmul(torch.t(inner_dir), inner_noise)/args.Q

            
            Jacobian -= args.tau * torch.t(h_xy + torch.matmul(h_yy, torch.t(Jacobian)))

            _, devx, devy = data_loader.get_batch_val(args.batch_size)
            dev_loss = args.crit_mean(inner(devx), devy)
            grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=inner.parameters())

            grad_lamda = torch.matmul(Jacobian, concat(grad_o_w))

            logger.update_err(dev_loss.data.cpu().numpy().item())
            # pdb.set_trace()
            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 


        elif args.alg == 'HFBiO_special':
            if hyt == 0:
                Jacobian = torch.zeros([5000, 7850]).cuda()
                noise_hyper_all = [[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in [ld,]] for _ in range(args.T)]
                noise_inner_all = [[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in inner.parameters()] for _ in range(args.T)]


            inner = inner_update(args, args.batch_size, inner, ld, data_loader)

            cnt, x, y = data_loader.get_batch_train(args.batch_size)
            loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(ld[cnt])) + args.lamda * inner.get_norm()
            grad_inner = concat(torch.autograd.grad(outputs=loss, inputs= inner.parameters()))

            noise_hyper = noise_hyper_all[hyt]
            hyper_params = [p + args.mu * n for p, n in zip([ld,], noise_hyper)]
            loss = torch.mean(args.crit(inner(x), y)*torch.sigmoid(hyper_params[0][cnt])) + args.lamda * inner.get_norm()
            grad_inner_hyper = concat(torch.autograd.grad(outputs=loss, inputs= inner.parameters()))
            grad_hyper_dir = (grad_inner_hyper - grad_inner) / args.mu

            hyper_noise = [concat(noise_hyper)]
            hyper_dir = [grad_hyper_dir]

            noise_inner = noise_inner_all[hyt]
            inner_params = [p + args.niu * n for p, n in zip(inner.parameters(), noise_inner)]
            inner_noise = Learner(config).cuda()
            for p, p_noise in zip(inner_params, inner_noise.parameters()):
                p_noise.data = p.data
            loss = torch.mean(args.crit(inner_noise(x), y)*torch.sigmoid(ld[cnt])) + args.lamda * inner_noise.get_norm()
            grad_inner_inner = concat(torch.autograd.grad(outputs=loss, inputs= inner_noise.parameters()))
            grad_inner_dir = (grad_inner_inner - grad_inner) / args.niu

            inner_noise = [concat(noise_inner)]
            inner_dir = [grad_inner_dir]
                
            hyper_noise = torch.stack(hyper_noise)
            hyper_dir = torch.stack(hyper_dir)

            inner_noise = torch.stack(inner_noise)
            inner_dir = torch.stack(inner_dir)


            h_xy = torch.matmul(torch.t(hyper_dir), hyper_noise)
            Jacobian -= args.tau * torch.t(h_xy + torch.matmul(torch.t(inner_dir), torch.matmul(inner_noise, torch.t(Jacobian))))

            # _, devx, devy = data_loader.get_batch_val(args.batch_size)
            devx, devy = data_loader.get_val()
            dev_loss = args.crit_mean(inner(devx), devy)
            grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=inner.parameters())

            grad_lamda = torch.matmul(Jacobian, concat(grad_o_w))

            logger.update_err(dev_loss.data.cpu().numpy().item())
            # pdb.set_trace()
            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())    


        elif args.alg == 'ESJ':
            if hyt == 0:
                noise_hyper_all = [[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in [ld,]] for _ in range(20000)]

            grad_lamda = torch.zeros_like(ld)

            inner = Learner(config).cuda()
            inner = inner_update(args, args.batch_size, inner, ld, data_loader)

            devx, devy = data_loader.get_val()
            dev_loss = args.crit_mean(inner(devx), devy)
            grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=inner.parameters())

            for q in range(args.Q):
                noise_hyper = noise_hyper_all[np.random.randint(20000)]
                hyper_params = [p + args.mu * n for p, n in zip([ld,], noise_hyper)]
                inner_hyper = Learner(config).cuda()
                inner_hyper = inner_update(args, args.batch_size, inner_hyper, hyper_params[0], data_loader)
                hyper_dir = (concat(inner_hyper.parameters()) - concat(inner.parameters())) / args.mu

                hyper_noise = concat(noise_hyper).view(1,-1)
                hyper_dir = hyper_dir.view(1,-1)

                grad_lamda_tmp = (torch.matmul(hyper_dir, concat(grad_o_w).view(-1,1)).item() * hyper_noise[0]).view(-1)
                grad_lamda = grad_lamda + grad_lamda_tmp.detach().clone()/args.Q

            logger.update_err(dev_loss.data.cpu().numpy().item())

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())    
            

        elif args.alg == 'HOZOJ':
            if hyt == 0:
                noise_hyper_all = [[torch.normal(torch.zeros(p.shape), torch.ones(p.shape)).cuda() for p in [ld,]] for _ in range(20000)]

            grad_lamda = torch.zeros_like(ld)

            inner = Learner(config).cuda()
            inner = inner_update(args, args.batch_size, inner, ld, data_loader)
            devx, devy = data_loader.get_val()
            dev_loss = args.crit_mean(inner(devx), devy).item()

            for q in range(args.Q):
                noise_hyper = noise_hyper_all[np.random.randint(20000)]
                hyper_params = [p + args.mu * n for p, n in zip([ld,], noise_hyper)]
                inner_hyper = Learner(config).cuda()
                inner_hyper = inner_update(args, args.batch_size, inner_hyper, hyper_params[0], data_loader)

                devx, devy = data_loader.get_val()
                dev_loss_hyper = args.crit_mean(inner_hyper(devx), devy).item()

                grad_lamda_tmp = (dev_loss_hyper - dev_loss) * noise_hyper[0] /args.mu
                grad_lamda = grad_lamda + grad_lamda_tmp.detach().clone()/args.Q

            logger.update_err(dev_loss)
            # print(dev_loss)

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item())    
        
        ###################################################################################################### 
        training_time = time.time() - start_time
        logger.update_time(training_time)                
        logger.update_f(compute_f1_score(ld, data_loader.y_index))
        # print(ld[:50])
        with torch.no_grad():
            testx, testy = data_loader.get_test()
            ans = torch.argmax(inner(testx),-1)
            correct = ans == testy
            acc = torch.mean(correct.float())
            logger.update_testAcc(acc.data.cpu().numpy().item())
        
        if hyt %  10==0: 
            logger.print(hyt)
            logger.save()

if __name__ == "__main__":
    # sending arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='VRBO', choices=['stocBiO', 'HOAG', 'TTSA', 'BSA', 'ESJ', 'HOZOJ', 'BiAdam', 'VR-BiAdam',
                                                        'reverse', 'AID_CG', 'AID_NS', 'VRBO', 'MRBO', 'MSTSA', 'STABLE', 'AsBio', 'FSLA', 'FSLA_ADA',
                                                         'SMB', 'SVRB', 'HFBiO', 'HFBiO_vanilla', 'HFBiO_special'])
    parser.add_argument('--outer_opt', type=str, default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--cr', type=str, default='CE', choices=['CE', 'SH'])
    parser.add_argument('--innerT', type=int, default= 1, help="Number of Inner Iters")
    parser.add_argument('--T', type=int, default=2000, help="Number of Outer Iters")
    parser.add_argument('--v_iter', type=int, default=3, help="Number of iterations to compute v")
    parser.add_argument('--spider_iters', type=int, default=3, help="Spider Frequency")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch_size")
    parser.add_argument('--spider_batch_size', type=int, default=5000, help="Spider Batch_size")
    parser.add_argument('--num_class', type=int, default=10, help="Number Class")

    parser.add_argument('--hlr', type=float, default=1, help="HyperLr")
    parser.add_argument('--lr', type=float, default= 0.01, help="InnerLr")
    parser.add_argument('--beta', type=float, default= 0.5, help="Shrinkage parameter used in Neumann series")
    parser.add_argument('--storm_coef', type=float, default= 1, help="Shrinkage parameter used in Neumann series")
    parser.add_argument('--beta_adam', type=float, default= 0.99, help="Exponetial Moving Average Coefficient")
    parser.add_argument('--lamda', type=float, default= 0.01, help="Regularization parameter for v using conjugate")

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

    #L1
    parser.add_argument('--th', type=float, default=0, help="")
    parser.add_argument('--l1_alpha', type=float, default=0, help="")
    parser.add_argument('--lasso', dest='lasso', action='store_true', help='')


    #FSLA_ADA
    parser.add_argument('--v_beta1', type=float, default= 0.9, help="Exponetial Moving Average Coefficient")
    parser.add_argument('--v_beta2', type=float, default= 0.99, help="Exponetial Moving Average Coefficient")

    parser.add_argument('--rho', type=float, default= 0.6, help="Noise Rate")
    parser.add_argument('--channel', type=int, default=1, help="Batch_size")
    parser.add_argument('--size', type=int, default=28, help="Batch_size")
    parser.add_argument('--num_sample', type=int, default=40000, help="Batch_size")

    args = parser.parse_args()

    print(args.lasso)
    
    prefix_dir =  '/ocean/projects/cis220038p/junyili/AdaBilevel'

    data_loader = Data(prefix_dir + '/processed_data/MNIST_data/', args.rho, args.size, args.channel)
    # data_loader = Data('./SVHN_data/')
    # data_loader = Data('./FMNIST_data/')
    # data_loader = Data('./CIFAR10_data/')
    # data_loader = Data('./QMNIST_data/')

    prefix = args.alg + '-' + args.outer_opt

    postfix = 'rho-' + str(args.rho) + '-T-' + str(args.T) + '-hlr-' + str(args.hlr) + '-innerT-' + str(args.innerT) + '-lr-' + str(args.lr) + '-bs-' + str(args.batch_size)
    if args.alg == 'AID_CG':
        postfix += '-v_iter-' + str(args.v_iter) + '-lamda-' + str(args.lamda) + '-beta-' + str(args.beta)
    elif args.alg == 'AID_NS':
        postfix += '-v_iter-' + str(args.v_iter) + '-beta-' + str(args.beta)
    elif args.alg == 'VRBO':
        postfix += '-spider_iters-' + str(args.spider_iters) + '-spider_bs-' + str(args.spider_batch_size)
    elif args.alg == 'AsBio':
        postfix += '-spider_iters-' + str(args.spider_iters) + '-spider_bs-' + str(args.spider_batch_size) + '-beta_adam-' + str(args.beta_adam) + '-s_coef-' + str(args.storm_coef)
    elif args.alg == 'MRBO':
        postfix += '-d-' + str(args.d) + '-m-' + str(args.m) + '-c_lamda-' + str(args.c_lamda) + '-c_inner-' + str(args.c_inner)
    elif args.alg == 'MSTSA':
        postfix +=  '-c_lamda-' + str(args.c_lamda)
    elif args.alg == 'SMB':
        postfix += '-c_lamda-' + str(args.c_lamda) + '-c_inner-' + str(args.c_inner)
    elif args.alg == 'HFBiO_vanilla':
        postfix += '-tau-' + str(args.tau) + '-' + args.cr
    elif args.alg == 'HFBiO_special':
        postfix += '-tau-' + str(args.tau) + '-mu-' + str(args.mu) + '-niu-' + str(args.niu)+  '-' + args.cr
    elif args.alg == 'HFBiO':
        postfix += '-Q-' + str(args.Q) + '-tau-' + str(args.tau) + '-mu-' + str(args.mu) + '-niu-' + str(args.niu)+ str(args.Q) + '-' + args.cr
    elif args.alg == 'ESJ':
        postfix +=  '-mu-' + str(args.mu) + '-Q-' + str(args.Q) + '-' + args.cr
    elif args.alg == 'HOZOJ':
        postfix +=  '-mu-' + str(args.mu) + '-Q-' + str(args.Q)+  '-' + args.cr
    elif args.alg == 'BiAdam':
        postfix +=  '-beta_adam-' + str(args.beta_adam) + '-s_coef-' + str(args.storm_coef)
    elif args.alg == 'VR-BiAdam':
        postfix +=  '-beta_adam-' + str(args.beta_adam) + '-s_coef-' + str(args.storm_coef)
    elif args.alg == 'FSLA':
        postfix += '-beta-' + str(args.beta)
    elif args.alg == 'FSLA_ADA':
        postfix += '-beta-' + str(args.beta) + '-beta1-' + str(args.v_beta1) + '-beta2-' + str(args.v_beta2)

    
    if args.lasso:
        postfix += '-th-' + str(args.th)

    if args.l1_alpha > 0:
        postfix += '-alpha-' + str(args.l1_alpha)
    
    print(postfix)
    logger = Logger(prefix_dir + '/data_cleaning_new2', prefix = prefix, postfix= postfix)

    
    train(args, data_loader, logger)
