import os
import random
import sys

import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import scipy.io as io
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from sklearn.metrics import roc_auc_score
import matplotlib.cm
import networkx as nx 
from sklearn.metrics import confusion_matrix

from network import *
from utils import *


class EvalHelper:
    def __init__(self, input_data_dims, feat, label, hyperpm, train_index, test_index):
        use_cuda = torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
        feat = torch.from_numpy(feat).float().to(dev)
        label = torch.from_numpy(label).long().to(dev)
        self.input_data_dims = input_data_dims
        self.dev = dev
        self.hyperpm = hyperpm
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.MF_mode = hyperpm.MF_mode
        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th
        self.feat = feat
        self.targ = label
        self.best_acc = 0
        self.best_acc_2 = 0
        self.MF_sav = tempfile.TemporaryFile()
        self.GCMP_sav = tempfile.TemporaryFile()
        num = train_index.shape[0]
        self.trn_idx = train_index



        self.val_idx = np.array(test_index)
        self.tst_idx = np.array(test_index)
        trn_label = label[self.trn_idx].cpu().numpy()
        val_label = label[self.val_idx].cpu().numpy()
        # tst_label = label[self.tst_idx].cpu().numpy()
        counter = Counter(trn_label)
        
        print(counter)

        self.dataLayer = InitialDataLayer(self.input_data_dims)
                
        self.MessagePassing0 = GFR(input_dim = input_data_dims, output_dim = input_data_dims, th = self.th, mode = self.GC_mode).to(dev)  

        weight = len(trn_label)/np.array(list(counter.values()))/self.n_class
        
        #self.out_dim = self.d_v * self.n_head + self.modal_num**2
        self.out_dim = self.d_v * self.n_head
        
        self.weight = torch.from_numpy(weight).float().to(dev)
        if self.MF_mode == 'concat':
            self.ModalFusion = PLN(input_data_dims, hyperpm).to(dev)
            
        self.optimizer_MP0 = optim.Adam(self.MessagePassing0.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        
        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.ModalFusion.apply(my_weight_init)
        
    def run_epoch(self, mode, iteration_MF=10, iteration_GC=10, end = ''):
        dev = self.dev
        if mode == 'pre-train':
            self.ModalFusion.train()
            self.GraphConstruct.eval()
            self.MessagePassing.eval()
            
            self.optimizer_MF.zero_grad()
            prob, hidden, attn = self.ModalFusion(self.feat)
            #cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            cls_loss = ClsLoss_noweight(prob, self.targ, self.trn_idx)
            cls_loss.backward()
            self.optimizer_MF.step()
            print('trn-loss-MF: %.4f' % cls_loss, end=' ')
            
        if mode == 'simple':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            self.MessagePassing.train()
            
            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            
            prob, fusion_feat, attn = self.ModalFusion(self.feat)
            
            adj = self.GraphConstruct(fusion_feat)
            graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity)
            
            normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
            prob, xx = self.MessagePassing(fusion_feat, normalized_adj)
            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            
            loss = cls_loss + graph_loss
            loss.backward()
            
            self.optimizer_MF.step()
            self.optimizer_GC.step()
            self.optimizer_MP.step()
            print('trn-loss-MF: %.4f trn-loss-GC: %.4f' % (cls_loss, graph_loss), end=' ')
            
        elif mode == 'simple-2':
            
            self.MessagePassing0.train()
            self.ModalFusion.train()
  
            self.optimizer_MP0.zero_grad()
            self.optimizer_MF.zero_grad()

            model_num, feat1 = self.dataLayer(self.feat)
            feat, adj, ALoss = self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)

            all_graph_loss = 0
            for i in range(model_num):  
                graph_loss = GraphConstructLoss(feat1[i], adj[i], self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity)
                all_graph_loss = all_graph_loss + graph_loss
            all_graph_loss = all_graph_loss/4
            
            prob,_,_ = self.ModalFusion(feat)

            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            
            loss = cls_loss + all_graph_loss
            
            loss.backward()
            
            self.optimizer_MP0.step()            
            self.optimizer_MF.step()
            
            print('trn-loss-MF: %.4f ' % cls_loss, end=' ')
            print('trn-loss-MF: %.4f trn-loss-MF: %.4f' % (cls_loss, ALoss), end=' ')
            
        elif mode == 'normal':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            self.MessagePassing.train()
            
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            loss_MF = 0
            loss_GC = 0
            #loss_GC = []
            
            for t in range(iteration_MF):
                self.optimizer_MF.zero_grad()
                
                prob, fusion_feat, attn = self.ModalFusion(self.feat)
                adj = self.GraphConstruct(fusion_feat)
                normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
                prob2, xx = self.MessagePassing(fusion_feat, normalized_adj)
                cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
                
                graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity)
                total_loss = cls_loss + graph_loss
                print('cls_loss: %.4f, graph_loss: %.4f' % (cls_loss, graph_loss))
                loss_MF += total_loss.item()
                total_loss.backward()
                self.optimizer_MF.step()
            
               
            for t in range(iteration_GC):
                self.optimizer_GC.zero_grad()
                self.optimizer_MP.zero_grad() 
                prob, fusion_feat, attn = self.ModalFusion(self.feat)
                adj = self.GraphConstruct(fusion_feat)
                adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
                prob, xx = self.MessagePassing(fusion_feat, adj)
                cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
                
                graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity)
                total_loss = cls_loss + graph_loss
                loss_GC += total_loss.item()
                total_loss.backward()
                self.optimizer_MP.step()
                self.optimizer_GC.step()
            
        elif mode == 'early_stop':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            self.MessagePassing.train()
            
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            loss_MF = 0
            loss_GC = 0
            #loss_GC = []
            
            for t in range(iteration_MF):
                self.optimizer_MF.zero_grad()
                
                prob, fusion_feat, attn = self.ModalFusion(self.feat)
                adj = self.GraphConstruct(fusion_feat)
                normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
                prob2, xx = self.MessagePassing(fusion_feat, normalized_adj)
                cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
                
                graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity)
                total_loss = cls_loss + graph_loss
                print('cls_loss: %.4f , : %.4f' % (cls_loss, graph_loss))
                loss_MF += total_loss.item()
                total_loss.backward()
                self.optimizer_MF.step()
                cur_acc = self.cal_acc(self.val_idx)
                if cur_acc > self.best_acc:
                    self.best_acc = cur_acc
                    self.MF_sav.close()
                    self.MF_sav = tempfile.TemporaryFile()
                    torch.save(self.ModalFusion.state_dict(), self.MF_sav)
            
            self.MF_sav.seek(0)
            self.ModalFusion.load_state_dict(torch.load(self.MF_sav))
            
            for t in range(iteration_GC):
                self.optimizer_GC.zero_grad()
                self.optimizer_MP.zero_grad() 
                prob, fusion_feat, attn = self.ModalFusion(self.feat)
                adj = self.GraphConstruct(fusion_feat)
                adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
                prob, xx = self.MessagePassing(fusion_feat, adj)    
                cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
                
                graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree, self.hyperpm.theta_sparsity)
                total_loss = cls_loss + graph_loss
                loss_GC += total_loss.item()
                total_loss.backward()
                self.optimizer_MP.step()
                self.optimizer_GC.step()
                cur_acc = self.cal_acc(self.val_idx)
                if cur_acc > self.best_acc_2:
                    self.best_acc_2 = cur_acc
                    self.GCMP_sav.close()
                    self.GCMP_sav = tempfile.TemporaryFile()
                torch.save([self.GraphConstruct.state_dict(),self.MessagePassing.state_dict()], self.GCMP_sav)
            
            self.GCMP_sav.seek(0)
            param_list = torch.load(self.GCMP_sav)
            self.GraphConstruct.load_state_dict(param_list[0])
            self.MessagePassing.load_state_dict(param_list[1])
    
    def print_trn_acc(self, mode = 'pre-train'):
        print('trn-', end='')
        trn_acc, trn_auc, targ_trn, pred_trn = self._print_acc(self.trn_idx, mode, end=' val-')
        val_acc, val_auc, targ_val, pred_val = self._print_acc(self.val_idx, mode)
        #print('pred:',pred_val[:10], 'targ:',targ_val[:10])
        return trn_acc, val_acc

    def print_tst_acc(self, mode = 'pre-train'):
        print('tst-', end='')
        tst_acc, tst_auc, targ_tst, pred_tst = self._print_acc(self.tst_idx, mode, tst = True)
        conf_mat = confusion_matrix(targ_tst.detach().cpu().numpy(), pred_tst.detach().cpu().numpy())
        return tst_acc, tst_auc, conf_mat

    def _print_acc(self, eval_idx, mode, tst = False, end='\n'):
        
        self.MessagePassing0.eval()
        self.ModalFusion.eval()
        
        model_num, feat1 = self.dataLayer(self.feat)
        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)
        
        if mode == 'pre-train':
            prob, _, attn = self.ModalFusion(feat)
            
        else:
            prob, fusion_feat, attn = self.ModalFusion(feat)
            
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
    
        auc = roc_auc_score(one_hot(targ, self.n_class).cpu().numpy(), one_hot(pred, self.n_class).cpu().numpy())
        print('auc: %.4f  acc: %.4f' % (auc, acc), end=end)
        if tst == True and mode != 'pre-train':
            
            print('attention maps have been saved.')
            np.save('./attn/attn_map_{}.npy'.format(self.hyperpm.datname), attn)
            np.savez('./graph/{}_{}_graph_'.format(self.hyperpm.datname, self.GC_mode), 
                     fused=fusion_feat.detach().cpu().numpy(),
                     label = self.targ.detach().cpu().numpy())
            
        return acc, auc, targ, pred
    
    def cal_acc(self, eval_idx):
        breakpoint()
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        prob, fusion_feat, attn = self.ModalFusion(self.feat)
        adj = self.GraphConstruct(fusion_feat)
        adj = normalize_adj(adj + torch.eye(adj.size(0)).to(self.dev))
        prob, xx = self.MessagePassing(fusion_feat, adj)
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        return acc
    
    def visualize(self, sav_prefix):
        
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()

        prob_MF, fusion_feat, attn = self.ModalFusion(self.feat)
        adj_o = self.GraphConstruct(fusion_feat)
        adj = normalize_adj(adj_o + torch.eye(adj_o.size(0)).to(self.dev))
        prob, xx = self.MessagePassing(fusion_feat, adj)
        
        prob = prob[self.val_idx]
        targ = self.targ[self.val_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
   
        g = nx.from_numpy_matrix(adj_o)
        n = self.feat.size(0)
        
        acc = [('.' if c else '?') for c in acc.astype(dtype=np.bool)]
        sets = np.zeros(n, dtype=np.float32)
        sets[self.trn_idx.cpu()] = 0
        sets[self.val_idx.cpu()] = 1
        sets[self.tst_idx.cpu()] = 2
        pos_gml = sav_prefix + '.gml'
        visualize_as_gdf(g, sav_prefix, list(range(n)), targ, pos_gml)
        visualize_as_gdf(g, sav_prefix + '_set', acc, sets, pos_gml)
        visualize_as_gdf(g, sav_prefix + '_trg', acc, targ, pos_gml)
        
        
        