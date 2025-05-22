import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch PML Training')
parser.add_argument('--backbone', type=str, default='RN101', 
                    choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], 
                    help='backbone type for CLIP model')

# Integrated dataset configuration
parser.add_argument('--data', default='coco', type=str, 
                    choices=['voc2007', 'voc2012', 'cub', 'coco'], 
                    help='dataset type')

# Dataset specific configurations will be set based on the selected dataset
dataset_configs = {
    'voc2007': {
        'num_classes': 20,
        'train_dataset': './dataset/VOC2007_category_train.txt',
        'val_dataset': './dataset/VOC2007_category_val.txt',
        'checkpoint': 'scinet+voc2007',
        'epochs': 120,
        'total_epochs': 120,
        'sparse_topk': 20,
        'lr': 4e-5,
        'weight_decay': 1e-4,
        'relation_file': 'relations/relation+voc2012.npy',
        'batch_size': 64,
        'log_file': 'scinet+voc2007.txt'
    },
    'voc2012': {
        'num_classes': 20,
        'train_dataset': './dataset/voc2012_train_singlelabel.txt',
        'val_dataset': './dataset/voc2012_val.txt',
        'checkpoint': 'scinet+voc2012',
        'epochs': 120,
        'total_epochs': 120,
        'sparse_topk': 20,
        'lr': 4e-5,
        'weight_decay': 1e-4,
        'relation_file': 'relations/relation+voc2012.npy',
        'batch_size': 64,
        'log_file': 'scinet+voc2012.txt'
    },
    'cub': {
        'num_classes': 312,
        'train_dataset': './dataset/cub_train_singlelabel.txt',
        'val_dataset': './dataset/cub_val.txt',
        'checkpoint': 'scinet+cub',
        'epochs': 30,
        'total_epochs': 30,
        'sparse_topk': 312,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'relation_file': 'relations/relation+cub.npy',
        'batch_size': 32,
        'log_file': 'scinet+cub.txt'
    },
    'coco': {
        'num_classes': 80,
        'train_dataset': './dataset/coco_train_singlelabel.txt',
        'val_dataset': './dataset/coco_train_singlelabel.txt',  # Using train file as default
        'checkpoint': 'scinet+coco',
        'epochs': 1,
        'total_epochs': 1,
        'sparse_topk': 62,
        'lr': 3e-5,
        'weight_decay': 1e-4,
        'relation_file': 'relations/relation+coco.npy',
        'batch_size': 32,
        'log_file': 'scinet+coco.txt'
    }
}

# Add dataset-specific parameters with default values from the config
parser.add_argument('--num_classes', type=int, help='number of classes in dataset')
parser.add_argument('--train_dataset', type=str, help='path to training dataset')
parser.add_argument('--val_dataset', type=str, help='path to validation dataset')
parser.add_argument('--checkpoint', type=str, help='checkpoint name')
parser.add_argument('--epochs', type=int, help='number of training epochs')
parser.add_argument('--total_epochs', type=int, help='total number of epochs')
parser.add_argument('--sparse_topk', type=int, help='sparse topk value')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--weight_decay', type=float, help='weight decay')
parser.add_argument('--relation_file', type=str, help='relation file path')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--log_file', type=str, help='log file path')

# General parameters
parser.add_argument('--label_proportion', default=0.5, type=float, help='label_proportion')
parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
parser.add_argument('--image_size', default=224, type=int, help='input image size')

# LearnablePrompts
parser.add_argument('--n_ctx', default=4, type=int, help='context length')
parser.add_argument('--ctx_init', default='a photo of a', type=str, help='context initialization')

# Loss
parser.add_argument('--loss', default='SPLC', type=str, help='loss function')
# consistency_loss
parser.add_argument('--p_cutoff', default=0.95, type=float, help='probability cutoff')
parser.add_argument('--kl_lambda', default=2, type=int, help='KL divergence lambda')
parser.add_argument('--hard_k', default=5, type=int, help='hard k value')
# total_loss = sup_loss + cfg.lambda_u * unsup_loss
parser.add_argument('--lambda_u', default=0.125, type=int, help='unsupervised loss weight')
parser.add_argument('--thre', default=0.5, type=float, help='threshold')

# Relation
parser.add_argument('--T', default=0.3, type=float, help='temperature')
parser.add_argument('--reweight_p', default=0.2, type=float, help='reweight probability')
parser.add_argument('--model_name', default='gcn', type=str, help='model name')
parser.add_argument('--gcn_lr', default=2e-4, type=float, help='GCN learning rate')
parser.add_argument('--ratio', default=1, type=int, help='get_ema_co')
parser.add_argument('--scale', default=10, type=int, help='scale factor')

# Testing and resuming
parser.add_argument('-t', '--test', help='run test', default=False, action="store_true")
parser.add_argument('-r', '--round', help='round number', default=1, type=int)
parser.add_argument('--resume', default=False, action='store_true', help='resume from checkpoint')

args = parser.parse_args()

# Set dataset-specific defaults if not explicitly provided
if args.data in dataset_configs:
    config = dataset_configs[args.data]
    for key, value in config.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

# Checkpoint directory handling
args.checkpoint = f"checkpoints/{args.checkpoint}"
dirs = os.listdir(args.checkpoint) if os.path.exists(args.checkpoint) else []
dirs.sort()
for i, dir in enumerate(dirs):
    assert (dir.startswith('round'))
next_index = len(dirs) + 1

if args.resume:
    args.resume = f"{args.checkpoint}/round{args.round}"
else:
    args.resume = False

if not args.test:
    args.checkpoint = f"{args.checkpoint}/round{next_index}"
    assert (not os.path.exists(args.checkpoint))
    args.test = False
else:
    args.checkpoint = f"{args.checkpoint}/round{args.round}"
    args.log_file = "test.txt"
    args.test = True

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)