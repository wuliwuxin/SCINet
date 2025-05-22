import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import transforms

from args import args
from log import logger
from model import SCINet, load_clip_model
from utils import COCO_missing_val_dataset, CocoDetection, PCocoDetection, ModelEma, get_ema_co

from randaugment import RandAugment


class ThreeTransfoamtions(torch.utils.data.Dataset):  # type: ignore

    def __init__(self, root, annFile, transform, target_transform=None, class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        self.name = names
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform
        self.strong_transform: transforms.Compose = copy.deepcopy(transform)
        self.extra_transform: transforms.Compose = copy.deepcopy(transform)
        self.strong_transform.transforms.insert(0, RandAugment(3))
        self.extra_transform.transforms.insert(0, RandAugment(1))

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        img_w = self.transform(img)
        img_s = self.strong_transform(img)
        img_e = self.extra_transform(img)

        return [index, img_w, img_s, img_e], label

    def __len__(self):
        return len(self.name)


def build_weak_strong_dataset(train_preprocess, val_preprocess, pin_memory=True):
    if "coco" in args.data:
        return build_coco_weak_strong_dataset(train_preprocess, val_preprocess)
    elif "voc2007" in args.data:
        return build_voc2007_weak_strong_dataset(train_preprocess, val_preprocess)

    elif "voc2012" in args.data:
        return build_voc2012_weak_strong_dataset(train_preprocess, val_preprocess)
    elif "cub" in args.data:
        return build_cub_weak_strong_dataset(train_preprocess, val_preprocess)
    else:
        assert (False)


def build_coco_weak_strong_dataset(train_preprocess, val_preprocess):

    instances_path_val = './dataset/coco/annotations/instances_val2014.json'

    instances_path_train = './dataset/coco_train_singlelabel.txt'

    data_path_val = './dataset/coco/val2014'
    data_path_train = './dataset/coco/train2014'

    val_dataset = CocoDetection(data_path_val, instances_path_val,
                                val_preprocess)
    train_dataset = ThreeTransfoamtions(data_path_train, instances_path_train, train_preprocess,
                                      class_num=args.num_classes)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    return [train_loader, val_loader]


# VOC2007
def build_voc2007_weak_strong_dataset(train_preprocess, val_preprocess):
    instances_path_train = args.train_dataset
    instances_path_val = args.val_dataset

    data_path_val = './dataset/VOC2007/JPEGImages'
    data_path_train = './dataset/VOC2007/JPEGImages'

    val_dataset = COCO_missing_val_dataset(data_path_val, instances_path_val, val_preprocess,
                                           class_num=args.num_classes)
    train_dataset = ThreeTransfoamtions(data_path_train, instances_path_train, train_preprocess,
                                        class_num=args.num_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)
    return [train_loader, val_loader]


# VOC2012
def build_voc2012_weak_strong_dataset(train_preprocess, val_preprocess):
    instances_path_train = args.train_dataset
    instances_path_val = args.val_dataset

    data_path_val = './dataset/VOC2012/JPEGImages'
    data_path_train = './dataset/VOC2012/JPEGImages'

    val_dataset = COCO_missing_val_dataset(data_path_val, instances_path_val, val_preprocess,
                                           class_num=args.num_classes)
    train_dataset = ThreeTransfoamtions(data_path_train, instances_path_train, train_preprocess,
                                        class_num=args.num_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)
    return [train_loader, val_loader]

# CUB
def build_cub_weak_strong_dataset(train_preprocess, val_preprocess):
    instances_path_train = args.train_dataset
    instances_path_val = args.val_dataset

    data_path_val = './dataset/cub/CUB_200_2011/images' 
    data_path_train = './dataset/cub/CUB_200_2011/images'

    val_dataset = COCO_missing_val_dataset(data_path_val, instances_path_val, val_preprocess,
                                           class_num=args.num_classes)
    train_dataset = ThreeTransfoamtions(data_path_train, instances_path_train, train_preprocess,
                                        class_num=args.num_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)
    return [train_loader, val_loader]


class SCINetTrainer():

    def __init__(self) -> None:
        super().__init__()

        clip_model, _ = load_clip_model()
        image_size = args.image_size

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        train_preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(), normalize
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), normalize
        ])

        train_loader, val_loader = build_weak_strong_dataset(
            train_preprocess,  # type: ignore
            val_preprocess)
        self.train_loader = train_loader
        self.val_loader = val_loader

        classnames = val_loader.dataset.labels()
        assert (len(classnames) == args.num_classes)

        self.model = SCINet(classnames, clip_model)
        self.relation = self.model.relation
        self.classnames = classnames
        for name, param in self.model.named_parameters():
            if "text_encoder" in name:
                param.requires_grad_(False)

        self.model.cuda()
        ema_co = get_ema_co()
        self.ema = ModelEma(self.model, ema_co)  # 0.9997^641=0.82

        self.selected_label = torch.zeros(
            (len(self.train_loader.dataset), args.num_classes),
            dtype=torch.long,
        )
        self.selected_label = self.selected_label.cuda()
        self.classwise_acc = torch.zeros((args.num_classes,)).cuda()
        self.classwise_acc[:] = 1 / args.num_classes

    def consistency_loss(self, logits_s, logits_w, y_lb):
        logits_w = logits_w.detach()

        pseudo_label = torch.sigmoid(logits_w)
        pseudo_label_s = torch.sigmoid(logits_s)

        relation_p = pseudo_label @ self.relation.cuda().t()

        max_probs, max_idx = torch.topk(pseudo_label, args.hard_k, dim=-1)

        threhold = args.p_cutoff * (self.classwise_acc[max_idx] /
                                    (2. - self.classwise_acc[max_idx]))

        mask = max_probs.ge(threhold).float().sum(dim=1) >= 1  # convex

        labels = torch.zeros((len(logits_s), args.num_classes), dtype=torch.long)
        for i, idx in enumerate(max_idx):
            labels[i][idx] = 1
        labels_mask = pseudo_label < args.p_cutoff * (
                self.classwise_acc / (2. - self.classwise_acc))
        labels[labels_mask] = 0
        labels = torch.logical_or(labels, y_lb.cpu()).type(torch.long)
        labels = labels.cuda()

        xs_pos = pseudo_label_s
        xs_neg = 1 - pseudo_label_s

        los_pos = labels * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - labels) * torch.log(xs_neg.clamp(min=1e-8))
        loss = (los_pos + los_neg) * mask.reshape(-1, 1)
        loss_kl = (relation_p * torch.log(xs_pos.clamp(min=1e-8)) + (1 - relation_p) * torch.log(
            xs_neg.clamp(min=1e-8))) * mask.reshape(-1, 1)

        return -loss.sum() - args.kl_lambda * loss_kl.sum(), labels

    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        x_ulb_idx, x_lb, x_ulb_w, x_ulb_s = input
        y_lb = target

        
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(), x_ulb_w.cuda(), x_ulb_s.cuda()
        x_ulb_idx = x_ulb_idx.cuda()

        pseudo_counter = self.selected_label.sum(dim=0)

        max_v = pseudo_counter.max().item()
        sum_v = pseudo_counter.sum().item()

        if max_v >= 1:  # not all(5w) -1
            for i in range(args.num_classes):
                self.classwise_acc[i] = max(pseudo_counter[i] / max(
                    max_v,
                    args.hard_k * len(self.selected_label) - sum_v), 1 / args.num_classes)

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        with autocast():
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

        logits_x_lb = logits_x_lb.float()
        logits_x_ulb_w, logits_x_ulb_s = logits_x_ulb_w.float(
        ), logits_x_ulb_s.float()

        sup_loss, _ = criterion(logits_x_lb, y_lb, epoch)

        unsup_loss, labels = self.consistency_loss(logits_x_ulb_s, logits_x_ulb_w, y_lb)

        assert (labels is not None)
        select_mask = labels.sum(dim=1) >= 1
        if x_ulb_idx[select_mask].nelement() != 0:
            self.selected_label[
                x_ulb_idx[select_mask]] = labels[select_mask]

        total_loss = sup_loss + args.lambda_u * unsup_loss

        return total_loss