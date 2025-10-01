# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import time
import cv2
import numpy as np
import logging
import argparse
from datetime import datetime  
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from PIL import Image
from model.panda import PANDA
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/metal/metal_fold0_swin.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger(args):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # �����ļ�������
    
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{args.split}-{args.base_lr}-{start_time}.txt"
    
    log_file_path = os.path.join(args.save_path, log_file_name)  # ���浽ָ��·��
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)  # ����Ŀ¼����������ڣ�?1�?7
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # �����ն˴�����
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # ������־��ʽ
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # ���Ӵ�������logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
    
def save_mask(model, val_loader, epoch):
    print(">>>>>>>>>>>> drawing masks >>>>>>>>>>>>>>>>")
    
    model.eval()
    device = next(model.parameters()).device  
    shot_type = "shot5" if val_loader.dataset.shot == 5 else "shot1"
    nshot = 5 if val_loader.dataset.shot == 5 else 1
    
    for batch_id, (input, target, s_input, s_mask, subcls, ori_label, query_img) in enumerate(val_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        s_input = s_input.to(device, non_blocking=True)
        s_mask = s_mask.to(device, non_blocking=True)
        ori_label = ori_label.to(device, non_blocking=True)

        with torch.no_grad():
            output= model(s_input,s_mask,input, target)
            output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=True)
            output = output.argmax(dim=1) 

        output = (output * 255).byte()

        date_str = datetime.datetime.now().strftime("%Y%m%d")
        output_dir = os.path.join(args.output_mask_dir,"PANDA", date_str,shot_type, f"{args.arch}_{args.dataname}_fold{args.split}_epoch{epoch}")
        os.makedirs(output_dir, exist_ok=True)


        for i in range(output.shape[0]):
            query_img_filename = os.path.basename(query_img[i])  
            query_img_base, _ = os.path.splitext(query_img_filename) 
            
            img_path = os.path.join(output_dir, f"{query_img_base}_mask.png")

      
            img = output[i].detach().cpu()  
            Image.fromarray(img.numpy()).save(img_path)
    print(f"the mask has been saved at {output_dir}")

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def print_optimizer_info(optimizer):
    if optimizer is None:
        print("Error: Optimizer is None!")
        return
    
    if not hasattr(optimizer, "param_groups") or not optimizer.param_groups:
        print("Error: optimizer.param_groups is None or empty!")
        return

    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}:")
        print(f"  LR: {group['lr']:.6f}")
        
        if 'params' not in group or not group['params']:
            print("  Warning: This param group has no parameters!")
            continue
        
        print(f"  Params count: {len(group['params'])}")
        
        try:
            param_names = [p.name[:20] + '...' for p in group['params'] if hasattr(p, 'name')]
            print(f"  Param names sample: {param_names if param_names else 'No named params'}")
        except Exception as e:
            print(f"  Error fetching param names: {e}")
        
        print("-" * 60)


def print_param_types(optimizer_params):
    for i, group in enumerate(optimizer_params):
        param_types = [type(p) for p in group['params']]
        print(f"Group {i}:")
        print(f"  Param types: {[t.__name__ for t in param_types]}")
        print(f"  Example param shape: {next(iter(group['params'])).shape}")
        print("-" * 60)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False  
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

 
    model = PANDA(multi_mode='fpn', output_size=(200, 200), classes=2,backbone= 'swin',pretain_path= "initmodel/swin_base_patch4_window12_384_22kto1k.pth")
    
    
    
    global logger, writer
    logger = get_logger(args)  
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    if not list(model.parameters()):
        print("Error: Model has no parameters! Check if it was initialized correctly.")
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if optimizer is None:
        print("Error: Optimizer initialization failed!")

    

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]
    train_transform = [
        transform.Resize(size=args.val_size),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
                                data_list=args.train_list, transform=train_transform, mode='train', \
                                )

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])    
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])           
        val_data_shot1 = dataset.SemData(split=args.split,
                                   shot=1,
                                   data_root=args.data_root,
                                   data_list=args.val_list,
                                   transform=val_transform,
                                   mode='val')
        val_sampler = None
        val_loader_shot1 = torch.utils.data.DataLoader(val_data_shot1,
                                                 batch_size=args.batch_size_val,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler)

        
    Re1_max_iou = 0.
    Re1_max_fbiou = 0
    Re1_best_epoch = 0
        

    filename = 'ours.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)     

        if args.evaluate:
            Re1_loss_val, Re1_mIoU_val, Re1_mAcc_val, Re1_allAcc_val, Re1_class_miou = validate(val_loader_shot1, model, criterion, nshot= 1)
            if main_process():
                
                writer.add_scalar('Re1_loss_val', Re1_loss_val, epoch_log)
                writer.add_scalar('Re1_mIoU_val', Re1_mIoU_val, epoch_log)
                writer.add_scalar('Re1_mAcc_val', Re1_mAcc_val, epoch_log)
                writer.add_scalar('Re1_class_miou_val', Re1_class_miou, epoch_log)
                writer.add_scalar('Re1_allAcc_val', Re1_allAcc_val, epoch_log)
            if Re1_class_miou > Re1_max_iou:
                save_mask(model,val_loader_shot1,epoch)
                Re1_max_iou = Re1_class_miou
                Re1_best_epoch = epoch
                if os.path.exists(filename):
                    os.remove(filename)
                filename = args.save_path +'/train_epoch_' + str(epoch) + '_' + str(Re1_max_iou) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           filename)

            if Re1_mIoU_val > Re1_max_fbiou:
                Re1_max_fbiou = Re1_mIoU_val
            logger.info('Re1, Best Epoch {:.1f}, Best IOU {:.4f} Best FB-IoU {:4F}'.format(Re1_best_epoch, Re1_max_iou, Re1_max_fbiou))



    filename = args.save_path +  f'/fold{args.split}_{args.epochs}_metal.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)                


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

  
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    vis_key = 0
    print('Warmup: {}'.format(args.warmup))
    for i, (input, target, s_input, s_mask, subcls,_) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader)//2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output, loss = model(s_input,s_mask,input, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        n = input.size(0)
      

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '                       
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))        
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion, nshot):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
  
    split_gap = 6
    
    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap  

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    if args.split != 999:
        test_num = 300 
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0    
    iter_num = 0
    total_time = 0
    for e in range(10):
        for i, (input, target, s_input, s_mask, subcls, ori_label,_) in enumerate(val_loader):
            if (iter_num-1) * args.batch_size_val >= test_num:
                break
            iter_num += 1    
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True) 
            ori_label = ori_label.cuda(non_blocking=True)
            start_time = time.time()

            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            out_mask = model.predict_mask_nshot(query_img= input, support_imgs = s_input, support_masks = s_mask, nshot=nshot)


            intersection, union, new_target = intersectionAndUnionGPU(out_mask, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
                
            subcls = subcls[0].cpu().numpy()[0]
            class_intersection_meter[(subcls-1)%split_gap] += intersection[1]
            class_union_meter[(subcls-1)%split_gap] += union[1] 

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num/100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i+1, class_iou_class[i]))            
    

    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()