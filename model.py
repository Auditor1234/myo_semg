import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict


class CNN(nn.Module):
    """
    input data shape must be shape(B,1,8,200)
    """
    def __init__(self, classes=10, features=False) -> None:
        super().__init__()
        self.features = features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,19), padding=(1,9)) # shape(B,32,8,200)
        self.pool1 = nn.MaxPool2d((1,10), stride=(1,10)) # shape(B,32,8,20)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=1) # shape(B,32,8,20)
        self.pool2 = nn.MaxPool2d((4,4), stride=(4,4)) # shape(B,32,2,5)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.drop = nn.Dropout()

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(320,64)

        self.linear2 = nn.Linear(64, classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))

        x = self.flat(x)
        x = self.drop(x)
        x = self.relu(self.linear1(x))

        if self.features:
            return x
        x = self.drop(x)
        x = self.softplus(self.linear2(x))
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float16))
        return ret.type(orig_type)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) # 768 12
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class EMGViT(nn.Module):
    # input shape(B,1,8,200)
    def __init__(self, output_dim, window_size=200, patch_size=8, width=64, layers=1, heads=32, features=False):
        super().__init__()
        self.features = features
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=False)

        scale = width ** -0.5 # 0.2
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((window_size // patch_size) + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        # shape(B,1,8,200)
        x = self.conv1(x)  # shape(B,width,1,25)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape(B,width,25)
        x = x.permute(0, 2, 1)  # shape(B,25,width)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape(B,25 + 1,width)
        x = x + self.positional_embedding.to(x.dtype) # shape(B,26,width)
        x = self.ln_pre(x) # shape(B,26,width)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x) # shape(26,B,width)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # shape(B,width)

        if self.features:
            return x

        if self.proj is not None:
            x = x @ self.proj
            x = self.softplus(x)

        return x # shape(1,512)


class VCConcat(nn.Module):
    def __init__(self, classes=10) -> None:
        super().__init__()

        feature_dim=64
        self.cnn = CNN(classes, features=True)
        self.vit = EMGViT(
                    window_size=200,
                    patch_size=8,
                    width=feature_dim,
                    layers=1,
                    heads=32,
                    output_dim=classes,
                    features=True
                    )
        self.linear = nn.Linear(feature_dim * 2, classes)
    
    def forward(self, x):
        x1 = self.vit(x)
        x2 = self.cnn(x)
        x = torch.concat((x1, x2), dim=-1)
        x = self.linear(x)
        return x


class VCEvidential(nn.Module):
    def __init__(self, classes=10) -> None:
        super().__init__()

        feature_dim=64
        self.classes = classes
        self.cnn = CNN(classes, features=True)
        self.vit = EMGViT(
                    window_size=200,
                    patch_size=8,
                    width=feature_dim,
                    layers=1,
                    heads=32,
                    output_dim=classes,
                    features=True
                    )
        self.linear1 = nn.Linear(feature_dim, classes)
        self.linear2 = nn.Linear(feature_dim, classes)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        x1 = self.softplus(self.linear1(self.vit(x)))
        x2 = self.softplus(self.linear2(self.cnn(x)))
        
        if self.training:
            return x1, x2
        
        from loss import DS_Combin
        return DS_Combin((x1 + 1, x2 + 1), self.classes)


class VCEnsemble(nn.Module):
    def __init__(self, classes=10) -> None:
        super().__init__()

        feature_dim=64
        self.cnn = CNN(classes)
        self.vit = EMGViT(
                    window_size=200,
                    patch_size=8,
                    width=feature_dim,
                    layers=1,
                    heads=32,
                    output_dim=classes
                    )
        self.weights = nn.Parameter(torch.rand(2))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x1 = self.vit(x)
        x2 = self.cnn(x)
        weights = self.softmax(self.weights)
        x = weights[0] * x1 + weights[1] * x2
        return x


class Gating(nn.Module):
    """
    input data shape must be shape(B,1,8,200)
    """
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(dropout_rate)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

        self.layer1 = nn.Linear(input_dim, 256)

        self.layer2 = nn.Linear(256, 128)

        self.layer3 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = self.flatten(x)
        x = self.softplus(self.layer1(x))
        x = self.drop(x)

        x = self.softplus(self.layer2(x))
        x = self.drop(x)

        x = self.layer3(x)
        x = self.softmax(x)

        return x

class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        
        # Assuming all experts have the same input dimension
        input_dim = 16 * 50
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)
  
        # Calculate the expert outputs
        outputs = torch.stack(
           [expert(x) for expert in self.experts], dim=2)

        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(outputs * weights, dim=2)


class MoE5(nn.Module):
    def __init__(self, trained_experts):
        super(MoE5, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        
        # shape(B,1,16,50)
        self.block = nn.Sequential(
            nn.Conv2d(1, 16, (3,11)), # shape(B,16,14,40)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 8)),
            nn.Dropout2d(),
            # shape(B,16,7,5)
            nn.Flatten(),
            # shape(B,16*7*5)
            nn.Linear(16*7*5, self.num_experts),
            nn.Softmax(dim=-1)
        )
        self.sampler = torch.distributions.normal.Normal(0, 1)
        self.weights = nn.Parameter(torch.ones(self.num_experts))

    def forward(self, x):
        # Get the weights from the gating network
        # weights = self.block(x.unsqueeze(1))
  
        # Calculate the expert outputs
        outputs = torch.stack(
           [expert(x) for expert in self.experts], dim=2)

        # Adjust the weights tensor shape to match the expert outputs
        weights = torch.softmax(self.weights, dim=-1)
        weights = weights.view(1, 1, -1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(outputs * weights, dim=2)


class NormedLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        return F.normalize(x,dim=1).mm(F.normalize(self.weight,dim=0))

class EMGTLC(nn.Module):
    def __init__(self, experts) -> None:
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)
        # self.attentions = nn.ModuleList([SpatialAttention() for _ in range(self.num_experts)])
        self.eta = 0.2
    
    def _normalize(self, x):
        # diff = torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0]
        # return (x - torch.min(x, dim=1, keepdim=True)[0]) / (diff + 1e-9)
        return F.normalize(x, dim=1)

    def forward(self, x):
        outs = []
        self.logits = outs
        b0 = None
        self.w = [torch.ones(len(x),dtype=torch.float,device=x.device)]
        self.save_info = []

        for i in range(self.num_experts):
            xi = self.experts[i](x)
            xi = self._normalize(xi)
            outs.append(xi)

            # evidential
            alpha = torch.exp(xi)+1
            S = alpha.sum(dim=1,keepdim=True)
            b = (alpha-1)/S
            u = xi.shape[1]/S.squeeze(-1)
            self.save_info.append(torch.cat([b, u.unsqueeze(1)], dim=1))

            # update w
            if b0 is None:
                C = 0
            else:
                bb = b0.view(-1,b0.shape[1],1)@b.view(-1,1,b.shape[1])
                C = bb.sum(dim=[1,2])-bb.diagonal(dim1=1,dim2=2).sum(dim=1)
            b0 = b
            self.w.append(self.w[-1]*u/(1-C))
            


        # dynamic reweighting
        # exp_w = torch.stack(self.w) / self.eta
        # exp_w = torch.softmax(exp_w[:-1], dim=0)
        # self.w = exp_w
        # exp_w = exp_w.unsqueeze(-1)
        exp_w = [torch.exp(wi/self.eta) for wi in self.w]
        exp_w = [wi/wi.sum() for wi in exp_w]
        exp_w = [wi.unsqueeze(-1) for wi in exp_w]

        reweighted_outs = [outs[i]*exp_w[i] for i in range(self.num_experts)]
        return sum(reweighted_outs)


class EMGBranch(nn.Module):
    def __init__(self, classes, num_experts) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.eta = 0.2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1s = nn.ModuleList([self._make_layer(16, 32) for _ in range(num_experts)])
        self.layer2s = nn.ModuleList([self._make_layer(32, 32) for _ in range(num_experts)])
        self.linears = nn.ModuleList([NormedLinear(32*12*3, classes) for _ in range(num_experts)])
        self.apply(self._weights_init)
    
    def _normalize(self, x):
        return F.normalize(x, dim=1)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5)),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(out_channels),
                nn.Dropout2d(),
                nn.AvgPool2d(kernel_size=(1, 3))
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        outs = []
        self.logits = outs
        b0 = None
        self.w = [torch.ones(len(x),dtype=torch.float,device=x.device)]
        self.save_info = []

        for i in range(self.num_experts):
            xi = self.layer1s[i](x)
            xi = self.layer2s[i](xi)
            xi = xi.flatten(1)
            xi = self.linears[i](xi)
            xi = self._normalize(xi)
            xi = xi * 15
            outs.append(xi)

            # evidential
            alpha = torch.exp(xi)+1
            S = alpha.sum(dim=1,keepdim=True)
            b = (alpha-1)/S
            u = xi.shape[1]/S.squeeze(-1)
            self.save_info.append(torch.cat([b, u.unsqueeze(1)], dim=1))

            # update w
            if b0 is None:
                C = 0
            else:
                bb = b0.view(-1,b0.shape[1],1)@b.view(-1,1,b.shape[1])
                C = bb.sum(dim=[1,2])-bb.diagonal(dim1=1,dim2=2).sum(dim=1)
            b0 = b
            self.w.append(self.w[-1]*u/(1-C))
            
        exp_w = [torch.exp(wi/self.eta) for wi in self.w]
        exp_w = [wi/wi.sum() for wi in exp_w]
        exp_w = [wi.unsqueeze(-1) for wi in exp_w]

        reweighted_outs = [outs[i]*exp_w[i] for i in range(self.num_experts)]
        return sum(reweighted_outs)


class EMGAttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x.shape(B,32,2,5)
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        ) # shape(1,16,2048)
        return x.squeeze(0)



class ViTEncoder(nn.Module):
    # input shape(B,1,16,50)
    def __init__(self, output_dim, width=64, layers=1, heads=32, features=False):
        super().__init__()
        self.features = features
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=(16, 2), stride=(16, 2), bias=False)

        scale = width ** -0.5 # 0.2
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((25) + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        # shape(B,1,8,50)
        x = self.conv1(x)  # shape(B,width,1,25)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape(B,width,25)
        x = x.permute(0, 2, 1)  # shape(B,25,width)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape(B,25 + 1,width)
        x = x + self.positional_embedding.to(x.dtype) # shape(B,26,width)
        x = self.ln_pre(x) # shape(B,26,width)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x) # shape(26,B,width)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # shape(B,width)

        if self.features:
            return x

        if self.proj is not None:
            x = x @ self.proj
            x = self.softplus(x)

        return x # shape(1,512)


class MoE5FC(nn.Module):
    def __init__(self, classes, num_experts=1, cls_num_list=None):
        super(MoE5FC, self).__init__()
        self.num_experts = num_experts
        m_list = torch.tensor(cls_num_list, dtype=torch.float, requires_grad=False)
        self.m_list = m_list / torch.max(m_list)
        
        # shape(B,1,16,50)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5)), # shape(B,32,14,46)
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Dropout2d(),
            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.AvgPool2d(kernel_size=(1, 3)), # shape(B,32,16,15)
            nn.Conv2d(32, 64, kernel_size=(3, 5)), # shape(B,64,12,11)
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Dropout2d(),
            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.AvgPool2d(kernel_size=(1, 3)), # shape(B,64,12,3)
            nn.Flatten(),
        )
        self.fc = nn.ModuleList([self.make_fc(classes) for _ in range(self.num_experts)])
    
    def to(self, device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        return self

    def make_fc(self, out_features):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(64*12*3, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        x = self.backbone(x.unsqueeze(1))
  
        # Calculate the expert outputs
        # if self.training:
        #     outputs = torch.stack([fc(x) - self.m_list[None, :] for fc in self.fc], dim=2)
        # else:
        outputs = torch.stack([fc(x) for fc in self.fc], dim=2)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.mean(outputs, dim=2)


class EMGBranchNaive(nn.Module):
    def __init__(self, classes, num_experts, dropout=0.2, reweight_epoch=30) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.classes = classes
        self.reweight_epoch = reweight_epoch
        self.share_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,5)),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(kernel_size=(1, 3)), # shape(B,32,16,15)
        )

        self.layer1s = nn.ModuleList([self._make_layer(32, 64, dropout) for _ in range(num_experts)])
        self.linears = nn.ModuleList([self._make_fc(64*12*3, classes) for _ in range(num_experts)])
        self.correct = None
        self.total = None
        self.pre_correct = None
        self.pre_total = None
        # self.linears = nn.ModuleList([self._make_normLinear(64*12*3, classes) for _ in range(num_experts)])
        self.eta = 1.5

    def _hook_before_epoch(self, epoch):
        if epoch >= self.reweight_epoch:
            self.pre_correct = torch.zeros(self.num_experts, self.classes)
            self.pre_total = torch.zeros(self.classes)

    def _hook_after_epoch(self, epoch):
        if self.pre_correct != None:
            self.correct = self.pre_correct
            self.total = self.pre_total
    
    def _normalize(self, x):
        return F.normalize(x, dim=1)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)

    def _make_layer(self, in_channels, out_channels, dropout):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5)),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(out_channels),
                nn.Dropout2d(dropout),
                nn.AvgPool2d(kernel_size=(1, 3))
            )
    
    def _make_fc(self, in_dim, classes):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def _make_normLinear(self, in_dim, classes):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            NormedLinear(64, classes),
        )
    
    def forward(self, x, y=None):
        x = x.unsqueeze(1)
        x = self.share_net(x)
        outs = []
        self.logits = outs
        b0 = None
        self.w = [torch.ones(len(x),dtype=torch.float,device=x.device)]

        if self.pre_total != None and not self.training:
            for i in range(len(y)):
                self.pre_total[y[i]] += 1

        for i in range(self.num_experts):
            xi = self.layer1s[i](x)
            xi = xi.flatten(1)
            xi = self.linears[i](xi)
            outs.append(xi)

            # correct predictions per class
            if self.pre_correct != None and not self.training:
                with torch.no_grad():
                    pred = torch.argmax(xi, dim=1)
                    for j in range(len(y)):
                        if y[j] == pred[j]:
                            self.pre_correct[i][y[j]] += 1

        #     # evidential
        #     alpha = torch.exp(xi)+1
        #     S = alpha.sum(dim=1,keepdim=True)
        #     b = (alpha-1)/S
        #     u = xi.shape[1]/S.squeeze(-1)

        #     # update w
        #     if b0 is None:
        #         C = 0
        #     else:
        #         bb = b0.view(-1,b0.shape[1],1)@b.view(-1,1,b.shape[1])
        #         C = bb.sum(dim=[1,2])-bb.diagonal(dim1=1,dim2=2).sum(dim=1)
        #     b0 = b
        #     self.w.append(self.w[-1]*u/(1-C))
        
        # exp_w = [torch.exp(wi/self.eta) for wi in self.w]
        # exp_w = [wi/wi.sum() for wi in exp_w]
        # exp_w = [wi.unsqueeze(-1) for wi in exp_w]

        # reweighted_outs = [outs[i]*exp_w[i] for i in range(self.num_experts)]
        # # return sum(reweighted_outs)
        # if self.correct != None:
        #     outs_weight = torch.softmax(self.correct, dim=0).to(x.device)
        #     reweighted_outs = [outs[i] * outs_weight[i] for i in range(self.num_experts)]
        #     return sum(reweighted_outs)
        # else:
        if self.correct != None:
            correct_sum = self.correct.sum(dim=1)
            outs_weight = (correct_sum / correct_sum.sum()).to(x.device)
            reweighted_outs = [outs[i] * outs_weight[i] for i in range(self.num_experts)]
            return sum(reweighted_outs)
        else:
            return sum(outs) / len(outs)
