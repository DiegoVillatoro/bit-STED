import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, Optional
from einops import einsum, rearrange
from typing import Optional, Tuple

#####################################################################################
################################## BIT lINEAR ##################################
def activation_quant(x: torch.Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization
    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-127, 127) / scale
    return y

#1bit quant
#def weight_quant(w: Tensor):
#    scale = w.abs().mean()
#    e = w.mean()
#    u = (w - e).sign() * scale
#    return u
#1.5bit quant
def weight_quant(w: torch.Tensor):
    gamma = w.abs().mean()
    e = w/(gamma+1e-5)
    u = torch.clamp(e.round(), min=-1, max=1)*gamma
    return u

class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        dim (int): The input dimension of the layer.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the layer.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        w = self.weight
        dim = self.in_features
        
        #SimpleRMSNorm
        self.scale = dim**-0.5
        x_norm = F.normalize(x, dim=-1) * self.scale

        # STE using detach
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        
        return y

######################################################################################
####################################### MHSA #########################################
######################################################################################
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_model, n_heads, kv_heads, dropout, device):
        super().__init__()

        assert n_model % n_heads == 0

        self.n_model = n_model
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = n_model // n_heads
        self.kv_model = n_model // n_heads * kv_heads
        
        self.fc_q = nn.Linear(n_model, n_model).to(device)
        self.fc_k = nn.Linear(n_model, self.kv_model).to(device)
        self.fc_v = nn.Linear(n_model, self.kv_model).to(device)

        self.fc_o = nn.Linear(self.kv_model, n_model).to(device)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, embedding size]
        #key = [batch size, key len, embedding size]
        #value = [batch size, value len, embedding size]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, embedding size]
        #K = [batch size, key len, embedding size]
        #V = [batch size, value len, embedding size]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        #print(Q.shape)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        #energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        num_head_groups = self.n_heads // self.kv_heads
        if num_head_groups > 1:
            # Separate the query heads into 'num_head_groups' chunks, and fold the group
            # dimension into the batch dimension.  This allows us to compute the attention
            # for each head in parallel, then sum over all of the groups at the end.
            #Q = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
            Q = Q.view(batch_size, self.kv_heads, num_head_groups, -1, self.head_dim).permute(0,2,1,3,4)
            #print("einsum")
            #print(Q.shape)
            #print(K.shape)
            #torch.Size([12, 2, 4, 16, 64])
            #torch.Size([12, 4, 16, 64])
            energy = einsum(Q, K, "b g h n d, b h s d -> b h n s")


        #energy = [batch size, n heads, query len, key len]

        attention = torch.softmax(energy/self.scale, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        #x = torch.matmul(self.dropout(attention), V)
        x = einsum(self.dropout(attention), V, "b h n s, b h s d -> b h n d")

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.kv_model)

        #x = [batch size, query len, embedding size]
        x = self.fc_o(x)

        #x = [batch size, query len, embedding size]

        return x, attention

######################################################################################
###################################### BitMGQA #######################################
######################################################################################
class BitMGQA(nn.Module):
    def __init__(self, n_model, n_heads, kv_heads, dropout, device):
        super().__init__()

        assert n_model % n_heads == 0

        self.n_model = n_model
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = n_model // n_heads
        self.kv_model = n_model // n_heads * kv_heads

        self.fc_q = BitLinear(n_model, n_model).to(device)
        self.fc_k = BitLinear(n_model, self.kv_model).to(device)
        self.fc_v = BitLinear(n_model, self.kv_model).to(device)
        
        self.fc_o = BitLinear(self.kv_model, n_model).to(device)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, embedding size]
        #key = [batch size, key len, embedding size]
        #value = [batch size, value len, embedding size]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, embedding size]
        #K = [batch size, key len, embedding size]
        #V = [batch size, value len, embedding size]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        #print(Q.shape)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        #energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        num_head_groups = self.n_heads // self.kv_heads
        if num_head_groups > 1:
            # Separate the query heads into 'num_head_groups' chunks, and fold the group
            # dimension into the batch dimension.  This allows us to compute the attention
            # for each head in parallel, then sum over all of the groups at the end.
            #Q = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
            Q = Q.view(batch_size, self.kv_heads, num_head_groups, -1, self.head_dim).permute(0,2,1,3,4)
            #print("einsum")
            #print(Q.shape)
            #print(K.shape)
            #torch.Size([12, 2, 4, 16, 64])
            #torch.Size([12, 4, 16, 64])
            energy = einsum(Q, K, "b g h n d, b h s d -> b h n s")


        #energy = [batch size, n heads, query len, key len]

        attention = torch.softmax(energy/self.scale, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        #x = torch.matmul(self.dropout(attention), V)
        x = einsum(self.dropout(attention), V, "b h n s, b h s d -> b h n d")

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.kv_model)

        #x = [batch size, query len, embedding size]
        x = self.fc_o(x)

        #x = [batch size, query len, embedding size]
        return x, attention

######################################################################################
####################################### MLP ##########################################
######################################################################################
class MLP(nn.Module):
    def __init__(self, n_ff, n_model, dropout=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_model, n_ff),
            nn.GELU(),
            nn.LayerNorm(n_ff),
            nn.Dropout(dropout),
            nn.Linear(n_ff, n_model)
        )
    def forward(self, x):
        return self.ff(x)

######################################################################################
###################################### Bit FF ########################################
######################################################################################
class BitFeedForward(nn.Module):
    def __init__(self, n_ff, n_model, dropout=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            BitLinear(n_model, n_ff, bias=True),
            nn.GELU(),
            
            nn.LayerNorm(n_ff),
            nn.Dropout(dropout),
            BitLinear(n_ff, n_model, bias=True)
        )
    def forward(self, x):
        return self.ff(x)
    
class PatchEmbedding2(nn.Module):
    def __init__(self, img_size, patch_size = 16, n_model = 512, in_channels = 1):
        super().__init__()
        self.grid_size = (img_size[0]//patch_size, img_size[1]//patch_size) # grid size, size of new image according to patch size
        self.num_patches = self.grid_size[0] * self.grid_size[1] # grid size x grid size
        self.conv = nn.Conv2d(in_channels, n_model, kernel_size=patch_size, stride=patch_size)
    def forward(self, X):
        #input shape X : batch_size, num of channels image, height, weight
        #output shape : batch_size, num_patches, size_embedding
        return self.conv(X).flatten(2).transpose(1,2)

class PatchEmbedding3(nn.Module):
    def __init__(self, img_size, patch_size = 16, n_model = 512, in_channels = 1):
        super().__init__()
        self.grid_size = (img_size[0]//patch_size, img_size[1]//patch_size) # grid size, size of new image according to patch size
        
        self.num_patches = self.grid_size[0] * self.grid_size[1] # grid size x grid size
        
        self.conv1 = nn.Conv2d(in_channels, n_model//4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n_model//4, n_model//2, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(n_model//2, n_model, kernel_size=2, stride=2)
        
        #self.feats = []
    def forward(self, X):
        #input shape X : batch_size, num of channels image, height, weight
        #output shape : batch_size, num_patches, size_embedding
        feat1 = self.conv1(X) #1//2 
        feat2 = self.conv2(feat1) #1//4
        feat3 = self.conv3(feat2) #1//8
        feats = []
        feats.append(feat1)
        feats.append(feat2)
        feats.append(feat3)
        return feat3.flatten(2).transpose(1,2), feats

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size = 16, n_model = 512, in_channels = 1):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size, img_size[1]//patch_size) # grid size, size of new image according to patch size
        self.num_patches = self.grid_size[0] * self.grid_size[1] # grid size x grid size
        
        self.ln1 = nn.LayerNorm(patch_size*patch_size*in_channels)
        self.ln2 = nn.LayerNorm(n_model)
        self.proj = nn.Linear(patch_size*patch_size*in_channels, n_model)
        self.activation = nn.GELU()
    def forward(self, X):
        #input shape X : batch_size, num of channels image, height, weight
        #output shape : batch_size, num_patches, size_embedding
        #print(X.shape)
        X = rearrange( X, "b c (ht hp) (wt wp) -> b (ht wt) (hp wp c)", hp=self.patch_size , wp=self.patch_size)
        #print(X.shape)
        return self.ln2(self.activation(self.proj(self.ln1(X))))
    
class Block_encoder(nn.Module):
    def __init__(self, n_model, mult, num_heads, dropout, device, bitNet):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_model)
        if bitNet:
            kv_heads = num_heads//2
            #self.attention = BitMGQA(n_model, num_heads, kv_heads)
            #self.ffn =  BitFeedForward(n_model, n_model, mult, post_act_ln=True, dropout=dropout)
            self.attention = BitMGQA(n_model, num_heads, kv_heads, dropout, device)
            self.ffn =  BitFeedForward(n_model*mult, n_model, dropout)
        else:
            kv_heads = num_heads//2
            self.attention = MultiHeadAttentionLayer(n_model, num_heads, kv_heads, dropout, device)
            #self.attention = nn.MultiheadAttention(n_model, num_heads)
            self.ffn = MLP(n_model*mult, n_model, dropout)
        self.ln2 = nn.LayerNorm(n_model)
        
        self.dropout = nn.Dropout(dropout)
        self.att = None
    
    def forward(self, X, mask=None):
        # X: batch_size, num_patches, size_embedding
        X_norm = self.ln1(X)
        att_output, att = self.attention(X_norm, X_norm, X_norm)
        self.att = att
        return torch.add( torch.add(X,self.dropout(att_output)), self.dropout(self.ffn(self.ln2(torch.add(X, self.dropout(att_output))))))
        #output batch_size, num_patches, size_embedding

class Encoder(nn.Module):
    def __init__(self, img_size, in_channels, n_model, mult, patch_size, num_heads=8, num_blks=1, blk_dropout=0.1, device='cpu', bitNet=True, features=False):
        super().__init__()
        self.size = img_size
        self.patch_size = patch_size
        self.features = features
        if patch_size==8 and features:
            self.patch_embedding1 = PatchEmbedding(img_size, 2, n_model//4, in_channels=in_channels)
            self.patch_embedding2 = PatchEmbedding((img_size[0]//2, img_size[1]//2), 2, n_model//2, in_channels=n_model//4)
            self.patch_embedding3 = PatchEmbedding((img_size[0]//4, img_size[1]//4), 2, n_model, in_channels=n_model//2)
            num_patches = 784
        else:
            self.patch_embedding = PatchEmbedding(img_size, patch_size, n_model, in_channels=in_channels)
            num_patches = self.patch_embedding.num_patches
        
        # Posicional embedding are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, n_model))
        #self.cls_token = nn.Parameter(torch.randn(1, 1 , n_model))
        #num_steps = self.patch_embedding.num_patches + num_classes #add cls token
        self.dropout = nn.Dropout(blk_dropout)
        #self.enc = Encoder(n_model, n_model*4, num_heads, num_blks, blk_dropout, device)
        self.ln = nn.LayerNorm(n_model)

        self.blks = nn.ModuleList([Block_encoder(n_model, mult,
                                                num_heads, blk_dropout, device, bitNet) for _ in range(num_blks)])
        
    def forward(self, X):
        # X: batch_size, N channels, height, weight
        if self.patch_size==8 and self.features:
            feats = []
            #X, feats = self.patch_embedding(X)
            X = self.patch_embedding1(X)
            
            B, _, n_model = X.size() 
            h, w = self.size[0]//2, self.size[1]//2
            #X = X[:,1:,:]#ignore first patch
            X = X.permute(0, 2, 1)
            X = X.contiguous().view(B, n_model, h, w)
            #print("x shape after emb1", X.shape)
        
            feats.append(X)
            X = self.patch_embedding2(X)
            
            B, _, n_model = X.size() 
            h, w = self.size[0]//4, self.size[1]//4
            #X = X[:,1:,:]#ignore first patch
            X = X.permute(0, 2, 1)
            X = X.contiguous().view(B, n_model, h, w)
            #print("x shape after emb2", X.shape)
            feats.append(X)
            X = self.patch_embedding3(X)
            #print("x shape after emb", X.shape)
            #feats.append(X)
            
        else:
            X = self.patch_embedding(X)
        
        # token: batch_size, 1, size_embedding
        #X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        
        # X: batch_size, num_patches+1, size_embedding
        X = self.dropout(X+self.pos_embedding)
        for blk in self.blks:
            X  = blk(X)
        X = self.ln(X)
        
        ############################################################################################
        ######################### Linear Projection of Flattened Patches ###########################
        ############################################################################################
        # reshape from (B, n_patch, n_model) to (B, n_model, h, w)
        B, n_patch, n_model = X.size() 
        h, w = self.size[0]//self.patch_size, self.size[1]//self.patch_size
        #X = X[:,1:,:]#ignore first patch
        X = X.permute(0, 2, 1)
        X = X.contiguous().view(B, n_model, h, w)
        
        if self.patch_size==8 and self.features:
            return X, feats
        else:
            return X