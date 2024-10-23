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

import pdb

def soft_th(X,threshold):
    res = []
    for p, thr in zip(X, threshold):
        p_np = p.data.cpu().numpy()
        new_p = np.sign(p_np) * np.maximum((np.abs(p_np) - thr.data.cpu().numpy()), np.zeros(p_np.shape))
        res.append(torch.tensor(new_p).float().cuda())
    return res

#Models used in the experiments
def convLayer(in_planes, out_planes, useDropout = False):
    "3x3 convolution with padding"
    seq = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    return seq

class INNER(nn.Module):
    def __init__(self, num_classes):
        super(INNER, self).__init__()
        self.layer1 = convLayer(3,64)
        self.layer2 = convLayer(64,64)
        self.layer3 = convLayer(64,64)
        self.layer4 = convLayer(64,64)
        self.fc = nn.Linear(256,num_classes)

        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

   
    def weights_init(self,module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def get_params(self, modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    yield m.weight
                    yield m.bias

    def getHyperRep_params(self):
        self.get_params([self.layer1, self.layer2, self.layer3])
    
    def getInner_params(self):
        self.get_params([self.layer4])
    

class Learner(nn.Module):
    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()


        self.config = config
        self.hyper_num = 16

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        # self.vars = []
        # self.vars_bn = []

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        # pdb.set_trace()

        # if vars is None:
        #     vars = self.vars
        # elif len(vars) == 2:
        #     vars.extend(list(self.vars[2:]))
        # elif len(vars) == 16:
        #     tmp_vars = list(self.vars[:2])
        #     tmp_vars.extend(vars)
        #     vars = tmp_vars


        if vars is None:
            vars = self.vars
        elif len(vars) == self.hyper_num:
            vars.extend(list(self.vars[self.hyper_num:]))
        elif len(vars) == 18 - self.hyper_num:
            tmp_vars = list(self.vars[:self.hyper_num])
            tmp_vars.extend(vars)
            vars = tmp_vars

        # pdb.set_trace()
        idx = 0
        bn_idx = 0
        # x = x.view(-1, 28 * 28)

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)


        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

    def getHyperRep_params(self):
        return self.vars[:self.hyper_num]
    
    def getInner_params(self):
        return self.vars[self.hyper_num:]
    
    def setHyperRep_params(self, new_params):
        for var, new_var in zip(self.vars[:self.hyper_num], new_params):
            var.data = new_var.data
        return




def get_Hv_times(args, new_params, model, lamda, x_spt, y_spt, t_num):
    new_params = [p.detach().clone().requires_grad_() for p in new_params]
    def hessian_vector_prod(v_merge):
        v = split_as_model(new_params, v_merge)
        logits = model(x_spt[t_num], new_params, bn_training=True)
        loss = F.cross_entropy(logits, y_spt[t_num])
        grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
        Hv = torch.autograd.grad(outputs=grad_i_w, inputs=new_params, grad_outputs= v)
        for g_p, pp in zip(Hv, v):
            g_p.add_(lamda * pp)
        Hv_merge = torch.cat([g.clone().detach().reshape(-1) for g in Hv]).reshape(-1)
        return Hv_merge
    return hessian_vector_prod


def inner_update(args, model, x_spt, y_spt, t_num, hyper_params = None, create_graph=False):
    if hyper_params is None:
        hyper_params = [p.detach().clone() for p in model.getHyperRep_params()]
    new_params = [p.detach().clone().requires_grad_() for p in model.getInner_params()]
    for _ in range(args.innerT):
        logits = model(x_spt[t_num], hyper_params + new_params, bn_training=True)
        loss = F.cross_entropy(logits, y_spt[t_num])
        grad = torch.autograd.grad(loss, new_params, create_graph=create_graph)
        new_params = list(map(lambda p: p[1] - args.lr * p[0], zip(grad, new_params)))
    return new_params


def grad_normal(args, model, x_spt, y_spt, t_num, create_graph=False, new_params = None):
    if new_params is None and create_graph:
        new_params = model.getInner_params()
    elif new_params is None:
        new_params = [p.detach().clone().requires_grad_() for p in model.getInner_params()]
    logits = model(x_spt[t_num], new_params, bn_training=True)
    loss = F.cross_entropy(logits, y_spt[t_num])
    grad = torch.autograd.grad(loss, new_params, create_graph=create_graph)
    return grad



def hyper_grad_cg(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num):
    new_params = [p.detach().clone().requires_grad_() for p in new_params]

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])

    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])

    grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=new_params)
    grad_o_w_merge = torch.cat([g.reshape(-1) for g in grad_o_w])

    v = []
    for p in new_params:
        v.append(torch.zeros_like(p).cuda())
    v_merge = torch.cat([vv.reshape(-1).clone() for vv in v])
    
    v_merge = cg_solve(get_Hv_times(args, new_params, model, args.lamda, x_spt, y_spt, t_num), grad_o_w_merge, x_init=v_merge, cg_iters=args.v_iter)
    v = split_as_model(new_params, v_merge)
    # pdb.set_trace()
    logits = model(x_spt[t_num], new_params, bn_training=True)
    loss = F.cross_entropy(logits, y_spt[t_num])
    grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
    grad_lamda = torch.autograd.grad(outputs=grad_i_w, inputs=model.getHyperRep_params(), grad_outputs= v)

    p_norm = 0
    for pp in model.getHyperRep_params():
        p_norm = p_norm + torch.norm(pp, p=1)
    grad_lamda = [g_o - g_l for g_o, g_l in zip(grad_o_ld, grad_lamda)]

    return dev_loss.data.cpu().numpy().item(), grad_lamda

def hyper_grad_ns(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num):
    new_params = [p.detach().clone().requires_grad_() for p in new_params]

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
    grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=new_params)

    v = []; p_vec = []
    for g_o in grad_o_w:
        p_vec.append(g_o.detach().clone())
        v.append(args.beta * g_o.detach().clone())
    
    for _ in range(args.v_iter):
        logits = model(x_spt[t_num], new_params, bn_training=True)
        loss = F.cross_entropy(logits, y_spt[t_num])
        grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
        grad_p = torch.autograd.grad(outputs=grad_i_w, inputs=new_params, grad_outputs= p_vec)                  
    
        for pp, vv, g_p in zip(p_vec, v, grad_p):
            pp -= args.beta * g_p.detach().clone()
            vv += args.beta *pp

    logits = model(x_spt[t_num], new_params, bn_training=True)
    loss = F.cross_entropy(logits, y_spt[t_num])
    grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
    grad_lamda = torch.autograd.grad(outputs=grad_i_w, inputs=model.getHyperRep_params(), grad_outputs= v)

    p_norm = 0
    for pp in model.getHyperRep_params():
        p_norm = p_norm + torch.norm(pp, p=1)
    grad_lamda = [g_o - g_l for g_o, g_l in zip(grad_o_ld, grad_lamda)]

    return dev_loss.data.cpu().numpy().item(), grad_lamda


def hyper_grad_dir(model, new_params, x_qry, y_qry, t_num):
    new_params = [p.detach().clone().requires_grad_() for p in new_params]

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

    return dev_loss.data.cpu().numpy().item(), grad_o_ld


def hyper_grad_fsla(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num, v_state):
    new_params = [p.detach().clone().requires_grad_() for p in new_params]

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
    grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=new_params)

    p_vec = []
    for g_o in grad_o_w:
        p_vec.append(g_o.detach().clone())
    
    for _ in range(args.v_iter):
        logits = model(x_spt[t_num], new_params, bn_training=True)
        loss = F.cross_entropy(logits, y_spt[t_num])
        grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
        grad_p = torch.autograd.grad(outputs=grad_i_w, inputs=new_params, grad_outputs= v_state)                     

        v_grad_norm = []
        for pp, vv, g_p in zip(p_vec, v_state, grad_p):
            v_grad = g_p.detach().clone() - pp.detach().clone()
            v_grad_norm.append(torch.norm(v_grad).detach().clone().cpu().numpy().item())
            vv -= args.beta * v_grad
        print(v_grad_norm)

    logits = model(x_spt[t_num], new_params, bn_training=True)
    loss = F.cross_entropy(logits, y_spt[t_num])
    grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
    grad_lamda = torch.autograd.grad(outputs=grad_i_w, inputs=model.getHyperRep_params(), grad_outputs= v_state)

    # ratio = [torch.norm(g_l)/torch.norm(g_o) for g_o, g_l in zip(grad_o_ld, grad_lamda)]
    # print('ratio:' , [r.detach().clone().cpu().numpy().item() for r in ratio])


    grad_lamda = [g_o - g_l for g_o, g_l in zip(grad_o_ld, grad_lamda)]

    return dev_loss.data.cpu().numpy().item(), grad_lamda, v_state, v_grad_norm


def hyper_grad_fsla_ada(args, hyt, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num, v_state, v_momentum, v_adaptive):
    new_params = [p.detach().clone().requires_grad_() for p in new_params]

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
    grad_o_ld = torch.autograd.grad(outputs=dev_loss, inputs=model.getHyperRep_params())

    logits_q = model(x_qry[t_num], new_params, bn_training=True)
    dev_loss = F.cross_entropy(logits_q, y_qry[t_num])
    grad_o_w = torch.autograd.grad(outputs=dev_loss, inputs=new_params)

    p_vec = []
    for g_o in grad_o_w:
        p_vec.append(g_o.detach().clone())
    
    for _ in range(args.v_iter):
        logits = model(x_spt[t_num], new_params, bn_training=True)
        loss = F.cross_entropy(logits, y_spt[t_num])
        grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
        grad_p = torch.autograd.grad(outputs=grad_i_w, inputs=new_params, grad_outputs= v_state)                     

        v_grad_norm = []
        for pp, vv, g_p, vv_m, vv_a in zip(p_vec, v_state, grad_p, v_momentum, v_adaptive):
            v_grad = g_p.detach().clone() - pp.detach().clone()
            if hyt > 0:
                vv_m.data = args.v_beta1 * vv_m + (1 - args.v_beta1) * v_grad
                vv_a.data = args.v_beta2 * vv_a + (1 - args.v_beta2) * v_grad**2
            else:
                vv_m.data = v_grad
                vv_a.data = v_grad**2           
            v_grad_norm.append(torch.norm(v_grad).detach().clone().cpu().numpy().item())
            vv -= args.beta * vv_m/(vv_a**0.5 + 1e-6)
        print(v_grad_norm)

    logits = model(x_spt[t_num], new_params, bn_training=True)
    loss = F.cross_entropy(logits, y_spt[t_num])
    grad_i_w = torch.autograd.grad(loss, new_params, create_graph=True)
    grad_lamda = torch.autograd.grad(outputs=grad_i_w, inputs=model.getHyperRep_params(), grad_outputs= v_state)

    # ratio = [torch.norm(g_l)/torch.norm(g_o) for g_o, g_l in zip(grad_o_ld, grad_lamda)]
    # print('ratio:' , [r.detach().clone().cpu().numpy().item() for r in ratio])

    grad_lamda = [g_o - g_l for g_o, g_l in zip(grad_o_ld, grad_lamda)]

    return dev_loss.data.cpu().numpy().item(), grad_lamda, v_state, v_momentum, v_adaptive, v_grad_norm


def concat(grad):
    return torch.cat([g.reshape(-1) for g in grad])

def inner_prod(tensor_list1, tensor_list2):
    
    res = 0
    for t1, t2 in zip(tensor_list1, tensor_list2):
        # pdb.set_trace()
        res += torch.dot(t1.reshape(-1), t2.reshape(-1))
    return res

def flip_parameters_to_tensors(module, state_dict, base):

    for key in module._parameters.keys():
        concat_key = base + '.' + key
        #print(concat_key)
        if concat_key in state_dict and ('weight' in concat_key or 'bias' in concat_key):
            #pdb.set_trace()
            module._parameters[key] = state_dict[concat_key]

    module_name = [k for k,v in module._modules.items()]

    if base is not '':
        base = base + '.'
    for name in module_name:
        flip_parameters_to_tensors(module._modules[name],state_dict,base+name)


class Data:
    def __init__(self,dire):
        train_path_x = dire + 'trainx.npy'; train_path_y = dire + 'trainy.npy'
        dev_path_x = dire + 'devx.npy'; dev_path_y = dire + 'devy.npy'
        test_path_x = dire + 'testx.npy'; test_path_y = dire + 'testy.npy'
        y_index_path = dire + 'y_index.npy'
        self.xs = np.load(train_path_x)
        self.ys = np.load(train_path_y)
        self.y_index = np.load(y_index_path)
        self.devx = torch.tensor(np.load(dev_path_x), dtype=torch.float).cuda()
        self.devy = torch.tensor(np.load(dev_path_y), dtype=torch.long).cuda()
        self.testx = torch.tensor(np.load(test_path_x), dtype=torch.float).cuda()
        self.testy = torch.tensor(np.load(test_path_y), dtype=torch.long).cuda()

    def get_batch_train(self, batch_size):
        cnt = np.random.choice(self.xs.shape[0], batch_size, replace=False)
        x = torch.tensor(self.xs[cnt], dtype=torch.float).cuda()
        y = torch.tensor(self.ys[cnt], dtype=torch.long).cuda()
        return cnt, x, y

    def get_batch_val(self, batch_size):
        cnt = np.random.choice(self.devx.shape[0], batch_size, replace=False)
        x = torch.tensor(self.devx[cnt], dtype=torch.float).cuda()
        y = torch.tensor(self.devy[cnt], dtype=torch.long).cuda()
        return cnt, x, y
    
    def get_val(self):
        return self.devx, self.devy
    
    def get_test(self):
        return self.testx, self.testy

class Logger_meta(object):
    def __init__(self, dire, prefix='', postfix=''):
        self.val_loss = []
        self.test_acc = []
        self.v_norm = []
        self.dire = dire
        self.prefix = prefix
        self.postfix = postfix
        self.running_time = []
        
    def update_err(self, err):   
        self.val_loss.append(err)
    
    def update_testAcc(self, acc):
        self.test_acc.append(acc)
    
    def update_time(self, time):
        self.running_time.append(time)

    def update_v_norm(self, vv_norm):
        self.v_norm.append(vv_norm)
        
    def save(self):
        np.save(self.dire + '/gho_err_'+ self.prefix + '-' + self.postfix, self.val_loss)
        np.save(self.dire + '/gho_acc_'+ self.prefix + '-' + self.postfix, self.test_acc)
        np.save(self.dire + '/gho_time_'+ self.prefix + '-' + self.postfix, self.running_time)
        np.save(self.dire + '/gho_vnorm_'+ self.prefix + '-' + self.postfix, self.v_norm)

    def print(self, hyt):
        print('Iter', hyt, ': ', "val err is: " , round(self.val_loss[-1],3),\
                              "test acc is: " , round(self.test_acc[-1],3), 'runing time is:' , round(self.running_time[-1], 3))


def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x


def split_as_model(new_params, tensor):
    t_split = []
    start = 0
    for p in new_params:
        step = len(p.reshape(-1))
        t_split.append(tensor[start:start+step].detach().clone().reshape(p.shape))
        start += step

    return t_split