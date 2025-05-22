import os
import random
from copy import deepcopy
from randaugment import RandAugment
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision import datasets as datasets
from torchvision import transforms
from args import args
from log import logger


def average_precision(output, target):
    epsilon = 1e-8

    indices = output.argsort()[::-1]
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0  # type: ignore
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class PCocoDetection(datasets.coco.CocoDetection):

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform

        self.strong_transform: transforms.Compose = deepcopy(transform)

        self.extra_transform: transforms.Compose = deepcopy(transform)

        self.strong_transform.transforms.insert(0, RandAugment(3))
        self.extra_transform.transforms.insert(0, RandAugment(1))

        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def labels(self):
        return [v["name"] for v in self.coco.cats.values()]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        img_w = self.transform(img)
        img_s = self.strong_transform(img)
        img_e = self.extra_transform(img)

        target = target.max(dim=0)[0]
        return [index, img_w, img_s, img_e], target


class CocoDetection(datasets.coco.CocoDetection):

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def labels(self):
        return [v["name"] for v in self.coco.cats.values()]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = target.max(dim=0)[0]
        return img, target


class COCO_missing_dataset(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 annFile,
                 transform=None,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()

        self.name = names

        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        if os.path.exists(os.path.join(self.root, path)) == False:
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype=torch.long)
            img = np.zeros((448, 448, 3))
            img = Image.fromarray(np.uint8(img))
            exit(1)
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        assert (self.target_transform is None)
        return [index, img], label

    def __len__(self):
        return len(self.name)

    def labels(self):
        if "coco" in args.data:
            assert (False)
        elif "nuswide" in args.data:
            with open('nuswide_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "voc" in args.data:
            with open('voc_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "cub" in args.data:
            with open('cub_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        else:
            assert (False)


class COCO_missing_val_dataset(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 annFile,
                 transform=None,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        self.name = names
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        if os.path.exists(os.path.join(self.root, path)) == False:
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype=torch.long)
            img = np.zeros((448, 448, 3))
            img = Image.fromarray(np.uint8(img))
            exit(1)
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        assert (self.target_transform is None)
        return img, label

    def __len__(self):
        return len(self.name)

    def labels(self):
        if "coco" in args.data:
            assert (False)
        elif "voc2007" in args.data:
            with open('dataset/labels/voc2007_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "voc2012" in args.data:
            with open('dataset/labels/voc2012_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "cub" in args.data:
            with open('dataset/labels/cub_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        else:
            assert (False)


class ModelEma(torch.nn.Module):

    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(),
                                      model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model,
                     update_fn=lambda e, m: self.decay * e +
                                            (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):

    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255),
                      random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    gcn = []
    gcn_no_decay = []
    prefix = "module." if torch.cuda.device_count() > 1 else ""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(f"{prefix}gc"):
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                gcn_no_decay.append(param)
            else:
                gcn.append(param)
            assert ("gcn" in args.model_name)
        elif len(param.shape) == 1 or name.endswith(
                ".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }, {
        'params': gcn_no_decay,
        'weight_decay': 0.
    }, {
        'params': gcn,
        'weight_decay': weight_decay
    }]


def get_ema_co():
    if "coco" in args.data:
        ema_co = np.exp(np.log(0.82) / (641 * args.ratio))
    elif "voc" in args.data:
        ema_co = np.exp(np.log(0.82) / (45 * args.ratio))
    elif "cub" in args.data:
        if args.batch_size == 96:
            ema_co = np.exp(np.log(0.82) / (63 * args.ratio))
        else:
            ema_co = np.exp(np.log(0.82) / (47 * args.ratio))
    else:
        assert (False)
    return ema_co
