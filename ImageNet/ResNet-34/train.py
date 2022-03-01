# -*-coding:utf-8-*-
import torch.utils
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torch.cuda.amp import autocast as autocast
import torchvision.models as models
import resnet34
import os, argparse, logging, sys
import random
import time
import torch.nn as nn
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
parser = argparse.ArgumentParser("birealnet18")
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--save', type=str, default='./newSave', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', default='/home/cqdx/ImageNet', help='path to dataset')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--T', '-temp', default=1, type=int, help='temperature controls soft degree (default: 1)')
parser.add_argument('--a', default=0.9, type=float, help='balance loss weight')
parser.add_argument('--sim', default=0.1, type=float, help='balance loss weight')
parser.add_argument('--l2', default=1e-2, type=float, help='distance coefficient')
parser.add_argument('--l12', default=2e-4, type=float, help='L12 regularization coefficient')
parser.add_argument('--cos_w', default=2e-2, type=float, help='cosine similarity coefficient')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnet34', help='path of ImageNet')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--type', default='torch.cuda.FloatTensor', help='type of tensor - e.g torch.cuda.FloatTensor')
parser.add_argument('--start_epoch', default=-1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--warm_up', dest='warm_up', action='store_true', default=True, help='use warm up or not')
args = parser.parse_args()
CLASSES = 1000

if not os.path.exists('log'):
    os.mkdir('log')

save_name = 'IE-Net_Res34'
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/{}.txt'.format(save_name)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info(args)


def cpt_tk(epoch):
    "compute t&k in back-propagation"
    T_min, T_max = torch.tensor(1e-2).float(), torch.tensor(1e1).float()
    Tmin, Tmax = torch.log10(T_min), torch.log10(T_max)
    t = torch.tensor([torch.pow(torch.tensor(10.), Tmin + (Tmax - Tmin) / args.epochs * epoch)]).float()
    k = max(1 / t, torch.tensor(1.)).float()
    return t, k


def main():
    random.seed(0)
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    # define the model
    model = resnet34.resnet34_1w1a()
    model = nn.DataParallel(model).cuda()

    logging.info(model)

    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    conv_param = (param for name, param in model.named_parameters() if ('fc' in name or 'conv' in name))
    param = (param for name, param in model.named_parameters() if not ('fc' in name or 'conv' in name))

    optimizer = torch.optim.SGD([{'params': conv_param, 'initial_lr': args.lr},
                                 {'params': param, 'initial_lr': args.lr, 'weight_decay': 0.}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # define the scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_up * 4, eta_min=0,
                                                           last_epoch=args.start_epoch)
    # Initialization
    best_top1_acc = 0

    # start_epoch = args.start_epoch + 1
    checkpoint_tar = os.path.join(args.save, '{}checkpoint.pth.tar'.format(save_name))
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        args.start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint {} epoch = {}".format(checkpoint_tar, checkpoint['epoch']))

    # adjust the learning rate according to the checkpoint
    for epoch in range(args.start_epoch + 1):
        scheduler.step()

    # load training data
    traindir = os.path.join('/home/cqu/ImageNet/', 'train')
    valdir = os.path.join('/home/cqu/ImageNet/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers)

    # * setup conv_modules
    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_modules.append(module)

    # train the model
    for epoch in range(args.start_epoch + 1, args.epochs):

        # *warm up
        if args.warm_up and epoch < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch + 1) / 5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])
        end = time.time()

        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        # * compute t/k in back-propagation
        t, k = cpt_tk(epoch)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.k = k.cuda()
                module.t = t.cuda()
        train_obj, train_top1_acc, train_top5_acc, weightlist = train(epoch, train_loader, model, criterion, optimizer)
        # * adjust Lr
        if epoch >= 4 * args.warm_up:
            scheduler.step()
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion)
        print("one epoch needs ", (time.time() - end))
        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer': optimizer.state_dict(),
            # 'amp': amp.state_dict(),
        }, is_best, args.save, save_name)

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
        target_var = target

        # compute outputy
        logits_student = model(images)

        loss = criterion(logits_student, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # images = images.cuda()
            # target = target.cuda()
            images = images.cuda()
            target = target.cuda()
            with autocast():
                # compute output
                logits = model(images)
                loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        logging.info(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
                     .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, save, save_name):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, '{}checkpoint.pth.tar'.format(save_name))
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, '{}model_best.pth.tar'.format(save_name))
        shutil.copyfile(filename, best_filename)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

