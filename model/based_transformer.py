import torch
from torch import nn

from einops import rearrange, repeat

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        # 后面条件为真，返回false，否则为true
        self.dropout = dropout
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    # 是否训练时采用dropkey策略
    def Attention(self,Q, K, V, use_DropKey, mask_ratio):
        atten = (Q * (Q.shape[1] ** -0.5)) @ K.transpose(-2, -1)
        # print(atten)

        if use_DropKey == True:
            m_r = torch.ones_like(atten) * mask_ratio
            atten = atten + torch.bernoulli(m_r) * -1e12

        atten = atten.softmax(dim=-1)
        x = atten @ V
        return x

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 训练时舍弃而在评价时不舍弃
        if self.training:
            out = self.Attention(q,k,v,True,self.dropout)
            # print(2)

        else :
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            out = torch.matmul(attn, v)



        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class based_transformer(nn.Module):
    def __init__(self, *,origin_channel, embedding_channel,feature, patch_size,num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.num_patches =  int(feature/patch_size)
        self.patch_embedding = nn.Conv1d(origin_channel,origin_channel*patch_size,kernel_size=patch_size,stride=patch_size)
        self.linner = nn.Linear(origin_channel*patch_size, dim)


        self.pos_embedding = nn.Parameter(torch.randn(1, (embedding_channel + 1), dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        #输出最终的分类数
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        x = self.patch_embedding(img)

        x = x.permute(0,2,1)
        x = self.linner(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]#加上位置编码
        x = self.dropout(x)

        #经过注意力机制和两层线性变换
        x = self.transformer(x)
        x_out = x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x),x_out


if __name__ == '__main__':
    v = based_transformer(origin_channel=192
        ,embedding_channel=128,feature=512,
        patch_size=8,
        num_classes = 14,
        dim = 128,
        depth = 3,
        heads = 8,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,pool='mean'
    )

    img = torch.randn(1, 192, 512)
    v.train()
    preds,a = v(img)   # (1, 1000)
    print(a.size())
