import torch
import torch.nn as nn
from functools import partial
import sys
import models.models_mae as models_mae
import os
from PIL import Image
import torchvision.transforms as T
import h5py
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
from models.GAT_cut import Guassian_AT_cut,Guassian_AT_para,Laplace_AT_cut,Guassian_AT_query_cut
import torch.nn.functional as F

class PromptGeneratorConv(nn.Module):
    def __init__(self,args,dropout = 0,kernel_size = 3):
        super().__init__()
        self.CrossAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True)
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        if args.G_copy_another:
            self.CrossAttention_SM = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print('dropout ',dropout)
        print('Conv\n')
        print('kernel_size ',kernel_size)
        self.conv_img = nn.Conv2d(1024,1024,kernel_size,1,(kernel_size-1)//2)
        self.conv_msk = nn.Conv2d(1024,1024,kernel_size,1,(kernel_size-1)//2)
        self.Layer_norm = nn.LayerNorm(1024)
        self.Linear = nn.Linear(1024,1024)
        self.args = args
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linear.weight)
        nn.init.zeros_(self.Linear.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # support_features  [B,N,98,1024]
    # query_features    [B,1,98,1024]
    # first self_attention 
    #       input       [B*N,98,1024]
    # second self_attention
    #       input       [B,49,1024]
    # third cross_attention
    #       input  k,v  [B*49,N,1024]
    #       input  q    [B*49,1,1024]
    #       get_attnweight v2 @ weight
    # fourth return     [B,196,1024]
    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        support_features = support_features.reshape(batchsize*N,7,14,1024)
        suppimg_features = support_features[:,:,:7,:]
        suppmask_features = support_features[:,:,7:,:]
        suppimg_features = suppimg_features.permute(0,3,1,2)
        suppmask_features = suppmask_features.permute(0,3,1,2)
        suppimg_features = self.conv_img(suppimg_features).permute(0,2,3,1)
        suppmask_features = self.conv_msk(suppmask_features).permute(0,2,3,1)
        support_features = torch.cat((suppimg_features,suppmask_features),dim=2)
        qss = support_features.reshape(batchsize*N,98,1024)
        if self.args.align_s:
            qss_layer_norm = self.Layer_norm(qss)
            ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
            support_features = qss + ats_ans #[B*N,98,1024]
        else :
            support_features = qss
        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        if self.args.align_q:
            qsq_layner_norm = self.Layer_norm(query_features_img)
            atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
            query_features_img = query_features_img + atq_ans #[B,49,1024]
        query_features_img = query_features_img.reshape(batchsize*49,1,1024)
        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]
        support_features_img = support_features_img.reshape(batchsize,N,49,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N,49,1024)
        support_features_img = support_features_img.permute(0,2,1,3).reshape(batchsize*49,N,1024)
        support_features_mask = support_features_mask.permute(0,2,1,3).reshape(batchsize*49,N,1024)
        attn_out1,attn_weight = self.CrossAttention_S(query_features_img,support_features_img,support_features_img)         #[B*49,1,1024]
        attn_out2 = (attn_weight @ (self.Linear(support_features_mask)))          #[B*49,1,1024]
        if self.args.G_copy_another:
            attn_out2,attn_weight = self.CrossAttention_SM(query_features_img,support_features_mask,support_features_mask)
        if self.args.G_only_div:
            attn_out1 = support_features_img.mean(dim=1, keepdim=True)
            attn_out2 = support_features_mask.mean(dim=1, keepdim=True)
        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)
        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        loss = 0
        if self.args.loss_choice == 'cos':
            cosine_loss_img = F.cosine_embedding_loss(query_features_img.reshape(batchsize*49,1024),attn_out1.reshape(batchsize*49,1024), torch.tensor([1]).to(self.args.device))
            cosine_loss_msk = F.cosine_embedding_loss(query_features_mask.reshape(batchsize*49,1024),attn_out2.reshape(batchsize*49,1024), torch.tensor([1]).to(self.args.device))
            loss = cosine_loss_img+cosine_loss_msk
        if self.args.loss_choice == 'l1':
            l1_loss_img = F.l1_loss(query_features_img, attn_out1)
            l1_loss_msk = F.l1_loss(query_features_mask, attn_out2)
            loss = l1_loss_img+l1_loss_msk
        if self.args.loss_choice == 'l2':
            l2_loss_img = F.mse_loss(query_features_img, attn_out1)
            l2_loss_msk = F.mse_loss(query_features_mask, attn_out2)
            loss = l2_loss_img + l2_loss_msk
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        loss = loss * self.args.lamba

        return canvas_tokens,loss



class PromptGeneratorCut(nn.Module):
    def __init__(self,dropout = 0,sigma = 0.01,device='cpu'):
        super().__init__()
        self.CrossAttention_S1 = Guassian_AT_cut(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True,sigma=sigma,device=device)
        # self.CrossAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 1,batch_first=True)
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print("Cut")
        print('dropout ',dropout)
        print('sigma',sigma)
        self.Linear = nn.Linear(1024,1024)
        self.Layer_norm = nn.LayerNorm(1024)
        self.sigma = sigma
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linear.weight)
        nn.init.zeros_(self.Linear.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # support_features  [B,N,98,1024]
    # query_features    [B,1,98,1024]
    # first self_attention 
    #       input       [B*N,98,1024]
    # second self_attention
    #       input       [B,49,1024]
    # third cross_attention
    #       input  k,v  [B*49,N,1024]
    #       input  q    [B*49,1,1024]
    #       get_attnweight v2 @ weight
    # fourth return     [B,196,1024]
    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        qss = support_features.reshape(batchsize*N,98,1024)
        qss_layer_norm = self.Layer_norm(qss)
        ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
        support_features = qss + ats_ans #[B*N,98,1024]

        ## support's self_attention to register

        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        qsq_layner_norm = self.Layer_norm(query_features_img)
        atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
        query_features_img = query_features_img + atq_ans #[B,49,1024]

        ## query's img self_attention to register

        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]

        support_features_img = support_features_img.reshape(batchsize,N*7*7,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N*7*7,1024)

        attn_out1,_,attn_weight = self.CrossAttention_S1(query_features_img,support_features_img,support_features_img,support_features_mask)
        attn_out2 = attn_weight @(self.Linear(support_features_mask))

        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)

        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        return canvas_tokens,0


class PromptGeneratorQueryCut(nn.Module):
    def __init__(self,dropout = 0,device='cpu'):
        super().__init__()
        self.CrossAttention_S1 = Guassian_AT_query_cut(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True,device=device)
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print("Cut")
        print('dropout ',dropout)
        self.linear = nn.Linear(49,1)
        self.Linear = nn.Linear(1024,1024)
        self.Layer_norm = nn.LayerNorm(1024)
        self.initialize_weights()
        self.sigmoid = nn.Sigmoid()


    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linear.weight)
        nn.init.zeros_(self.Linear.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # support_features  [B,N,98,1024]
    # query_features    [B,1,98,1024]
    # first self_attention 
    #       input       [B*N,98,1024]
    # second self_attention
    #       input       [B,49,1024]
    # third cross_attention
    #       input  k,v  [B*49,N,1024]
    #       input  q    [B*49,1,1024]
    #       get_attnweight v2 @ weight
    # fourth return     [B,196,1024]
    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        qss = support_features.reshape(batchsize*N,98,1024)
        qss_layer_norm = self.Layer_norm(qss)
        ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
        support_features = qss + ats_ans #[B*N,98,1024]

        ## support's self_attention to register

        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        qsq_layner_norm = self.Layer_norm(query_features_img)
        atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
        query_features_img = query_features_img + atq_ans #[B,49,1024]

        ## query's img self_attention to register
        avg_features = query_features_img.mean(dim=2)
        linear_output = self.linear(avg_features)
        Sigma = self.sigmoid(linear_output).squeeze(dim=1)

        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]

        support_features_img = support_features_img.reshape(batchsize,N*7*7,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N*7*7,1024)

        attn_out1,_,attn_weight = self.CrossAttention_S1(query_features_img,support_features_img,support_features_img,support_features_mask,Sigma)
        attn_out2 = attn_weight @(self.Linear(support_features_mask))

        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)

        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        return canvas_tokens,0

class PromptGeneratorLaplaceCut(nn.Module):
    def __init__(self,dropout = 0,sigma = 0.01,device='cpu'):
        super().__init__()
        self.CrossAttention_S1 = Laplace_AT_cut(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True,sigma=sigma,device=device)
        # self.CrossAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 1,batch_first=True)
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print("Cut")
        print('dropout ',dropout)
        print('sigma',sigma)
        self.Linear = nn.Linear(1024,1024)
        self.Layer_norm = nn.LayerNorm(1024)
        self.sigma = sigma
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linear.weight)
        nn.init.zeros_(self.Linear.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # support_features  [B,N,98,1024]
    # query_features    [B,1,98,1024]
    # first self_attention 
    #       input       [B*N,98,1024]
    # second self_attention
    #       input       [B,49,1024]
    # third cross_attention
    #       input  k,v  [B*49,N,1024]
    #       input  q    [B*49,1,1024]
    #       get_attnweight v2 @ weight
    # fourth return     [B,196,1024]
    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        # print(support_features.shape)
        qss = support_features.reshape(batchsize*N,98,1024)
        qss_layer_norm = self.Layer_norm(qss)
        ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
        support_features = qss + ats_ans #[B*N,98,1024]

        ## support's self_attention to register

        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        qsq_layner_norm = self.Layer_norm(query_features_img)
        atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
        query_features_img = query_features_img + atq_ans #[B,49,1024]

        ## query's img self_attention to register

        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]

        support_features_img = support_features_img.reshape(batchsize,N*7*7,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N*7*7,1024)

        attn_out1,_,attn_weight = self.CrossAttention_S1(query_features_img,support_features_img,support_features_img,support_features_mask)
        attn_out2 = attn_weight @(self.Linear(support_features_mask))

        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)

        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        return canvas_tokens,0

class PromptGeneratorPara(nn.Module):
    def __init__(self,dropout = 0,sigma = 0.01,device='cpu'):
        super().__init__()
        self.CrossAttention_S1 = Guassian_AT_para(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True,sigma=sigma,device=device)
        # self.CrossAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 1,batch_first=True)
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print("Para")
        print('dropout ',dropout)
        print('sigma',sigma)
        self.Linear = nn.Linear(1024,1024)
        self.Layer_norm_s = nn.LayerNorm(1024)
        self.Layer_norm_q = nn.LayerNorm(1024)
        self.sigma = sigma
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linear.weight)
        nn.init.zeros_(self.Linear.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # support_features  [B,N,98,1024]
    # query_features    [B,1,98,1024]
    # first self_attention 
    #       input       [B*N,98,1024]
    # second self_attention
    #       input       [B,49,1024]
    # third cross_attention
    #       input  k,v  [B*49,N,1024]
    #       input  q    [B*49,1,1024]
    #       get_attnweight v2 @ weight
    # fourth return     [B,196,1024]
    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        qss = support_features.reshape(batchsize*N,98,1024)
        qss_layer_norm = self.Layer_norm_s(qss)
        ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
        support_features = qss + ats_ans #[B*N,98,1024]
        ## support's self_attention to register
        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        qsq_layner_norm = self.Layer_norm_q(query_features_img)
        atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
        query_features_img = query_features_img + atq_ans #[B,49,1024]

        ## query's img self_attention to register

        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]

        support_features_img = support_features_img.reshape(batchsize,N*7*7,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N*7*7,1024)

        attn_out1,_,attn_weight = self.CrossAttention_S1(query_features_img,support_features_img,support_features_img,support_features_mask)
        attn_out2 = attn_weight @(self.Linear(support_features_mask))

        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)

        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        return canvas_tokens,0

def laplace_weight_matrix(size, center, b=0.4):
    """
    Create a 2D Laplace weight matrix with the specified size and b, centered at the given location.
    """
    x = torch.linspace(0, size - 1, size)
    y = torch.linspace(0, size - 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(- (torch.abs(xx - center[0]) + torch.abs(yy - center[1])) / b)
    return kernel / kernel.sum()


def gaussian_weight_matrix(size, center, sigma=1.0):
    """
    Create a 2D Gaussian weight matrix with the specified size and sigma, centered at the given location.
    """
    x = torch.linspace(0, size - 1, size)
    y = torch.linspace(0, size - 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))
    return kernel / kernel.sum()

class Matrix():
    def __init__(self,sigma,device):
        self.gw = [[] for i in range(7)]
        for i in range(7):
            for j in range(7):
                self.gw[i].append(gaussian_weight_matrix(7, center=(i, j), sigma=sigma).to(device=device))
    def calculate_attention(self,key, value1, value2, query, dropout, tsigma=1.0):
        B, N, _, _, D = key.size()
        _, h, w, _ = query.size()

        # Reshape query to 49 x B x D
        query_reshaped = query.view(B, -1, D).permute(1, 0, 2)

        list1 = []
        list2 = []
        key = key.to(torch.float64)
            
        for i in range(h * w):
            # Calculate the position of the current patch
            patch_y, patch_x = divmod(i, w)
            # Current query vector
            q = query_reshaped[i]  # B x D
            q = q.to(torch.float64)
            # Compute similarity scores between query and key
            score = torch.einsum('bd,bnwhd->bnwh', q, key)  # B x N x 7 x 7
            # Create a Gaussian weight matrix centered at the current patch position
            gaussian_matrix = self.gw[patch_y][patch_x]
            score = score / torch.sqrt(torch.tensor(D))
            # Apply Gaussian weighting
            score = score * gaussian_matrix
            attn_weight = F.softmax(score.view(B, -1), dim=-1).view(B, N, 7, 7).to(torch.float32)
            # Compute weighted sum
            attn_output1 = torch.einsum('bnwh,bnwhd->bd', attn_weight, value1)
            attn_output2 = torch.einsum('bnwh,bnwhd->bd', attn_weight, value2)
            # Collect the result
            list1.append(attn_output1)
            list2.append(attn_output2)


        output1 = torch.stack(list1,dim=1)
        output2 = torch.stack(list2,dim=1)

        return output1, output2

class PromptGenerator(nn.Module):
    def __init__(self,dropout = 0,sigma = 1.0,device='cpu'):
        super().__init__()
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print('dropout ',dropout)
        print('sigma',sigma)
        self.Layer_norm = nn.LayerNorm(1024)
        self.Linearq = nn.Linear(1024,1024)
        self.Lineark = nn.Linear(1024,1024)
        self.Linearv1 = nn.Linear(1024,1024)
        self.Linearv2 = nn.Linear(1024,1024)
        self.dropout = nn.Dropout(0.25)
        self.Matrix = Matrix(sigma,device=device)
        self.sigma = sigma
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linearv2.weight)
        nn.init.zeros_(self.Linearv2.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # support_features  [B,N,98,1024]
    # query_features    [B,1,98,1024]
    # first self_attention 
    #       input       [B*N,98,1024]
    # second self_attention
    #       input       [B,49,1024]
    # third cross_attention
    #       input  k,v  [B*49,N,1024]
    #       input  q    [B*49,1,1024]
    #       get_attnweight v2 @ weight
    # fourth return     [B,196,1024]
    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        qss = support_features.reshape(batchsize*N,98,1024)
        qss_layer_norm = self.Layer_norm(qss)
        ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
        support_features = qss + ats_ans #[B*N,98,1024]

        ## support's self_attention to register

        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        qsq_layner_norm = self.Layer_norm(query_features_img)
        atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
        query_features_img = query_features_img + atq_ans #[B,49,1024]

        ## query's img self_attention to register

        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]
        support_features_img = support_features_img.reshape(batchsize,N,7,7,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N,7,7,1024)
        ## update this shape matching

        attn_out1,attn_out2 = self.Matrix.calculate_attention(self.Lineark(support_features_img),self.Linearv1(support_features_img),self.Linearv2(support_features_mask),self.Linearq(query_features_img),self.dropout,tsigma=self.sigma)

        ##after attention process the answer

        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)
        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        return canvas_tokens
    