from collections import OrderedDict
from timm.models.vision_transformer import resize_pos_embed
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from modified_clip import modified_clip
from modified_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from args import args
from log import logger

_tokenizer = _Tokenizer()

def load_clip_to_cpu():

    assert args.backbone in ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], \
        f"backbone must be one of ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16']"
    
    url = modified_clip._MODELS[args.backbone]
    model_path = modified_clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(  # type: ignore
            model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = modified_clip.build_model(state_dict or model.state_dict())  # type: ignore

    return model


class TextDominantEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class LearnablePrompts(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        args_imsize = 224
        assert args_imsize == clip_imsize, f"args_imsize ({args_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = args.ctx_init.replace("_", " ")
        assert (n_ctx == len(ctx_init.split(" ")))
        prompt = modified_clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([modified_clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])
        self.register_buffer("token_middle", embedding[:, 1:(1 + n_ctx), :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],  # type: ignore
            dim=1,
        )
        return prompts


def load_clip_model():
    clip_model = load_clip_to_cpu()

    clip_model.float()
    return clip_model, modified_clip._transform(clip_model.visual.input_resolution)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SCINet(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.learnable_prompt = LearnablePrompts(classnames, clip_model)
        self.tokenized_prompts = self.learnable_prompt.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextDominantEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.use_spectral_norm = False
        
        if args.backbone == 'RN50':
            self.gc1 = GraphConvolution(1024, 2048)
            self.gc2 = GraphConvolution(2048, 2048)
            self.gc3 = GraphConvolution(2048, 1024)
        else:  # 'RN101', 'ViT-B/32', 'ViT-B/16'
            self.gc1 = GraphConvolution(512, 1024)
            self.gc2 = GraphConvolution(1024, 1024)
            self.gc3 = GraphConvolution(1024, 512)

        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)

        self.relation = torch.Tensor(np.load(args.relation_file))
        _, max_idx = torch.topk(self.relation, args.sparse_topk)
        mask = torch.ones_like(self.relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        self.relation[mask] = 0
        sparse_mask = mask

        dialog = torch.eye(args.num_classes).type(torch.bool)
        self.relation[dialog] = 0

        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * args.reweight_p
        self.relation[dialog] = 1 - args.reweight_p

        self.gcn_relation = self.relation.clone()
        assert (self.gcn_relation.requires_grad == False)

        self.relation = torch.exp(self.relation / args.T) / torch.sum(torch.exp(self.relation / args.T), dim=1).reshape(
            -1, 1)
        self.relation[sparse_mask] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1)

    def forward(self, image):

        tokenized_prompts = self.tokenized_prompts

        image_features = self.image_encoder(image.type(self.dtype))

        image_features_fro_norm = image_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features_fro_norm

        logit_scale = self.logit_scale.exp()

        if args.scale != 'modified_clip':
            assert (isinstance(args.scale, int))
            logit_scale = args.scale

        prompts = self.learnable_prompt()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        identity = text_features

        if self.use_spectral_norm:
            text_features_spec_norm = torch.linalg.norm(text_features, ord=2, dim=-1, keepdim=True)
            text_features = text_features / text_features_spec_norm
        else:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features = self.gc1(text_features, self.gcn_relation.cuda())
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, self.gcn_relation.cuda())
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, self.gcn_relation.cuda())

        text_features += identity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        return logits
