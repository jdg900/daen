import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import random
import wandb
from tqdm import tqdm
from datetime import datetime

from model.refinenetlw_hsm import rf_lw101
from utils.losses import CrossEntropy2d
from dataset.cityscapes_dataset import cityscapesDataset
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = 'without_pretraining'


def init_ema_weights(model, ema_model):
    for param in ema_model.parameters():
            param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    
    for i in range(0, len(mp)):
        if not mcp[i].data.shape:  # scalar tensor
            mcp[i].data = mp[i].data.clone()
        else:
            mcp[i].data[:] = mp[i].data[:].clone()


def update_ema_weights(model, ema_model, alpha, iter):
    alpha = min(1 - 1 / (iter + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if not param.data.shape:
            ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data
        else:
            ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]


def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)


def loss_calc_un(pred, label, gpu, pixel_weight):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    loss = criterion(pred, label)
    loss = loss * pixel_weight.to(loss.dtype)
    return torch.mean(loss)


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers

def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

def main():
    """Create the model and start the training."""

    args = get_arguments()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'

    wandb.init(project='iccv23', name=f'{run_name}', )
    wandb.config.update(args)

    cudnn.enabled = True
    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        model = rf_lw101(num_classes=args.num_classes)
        print(RESTORE_FROM)
    else:
        restore = torch.load(args.restore_from)
        model = rf_lw101(num_classes=args.num_classes)
        model.load_state_dict(restore['state_dict'])
        start_iter = 0

    ema_model = rf_lw101(num_classes=args.num_classes)

    # initialize ema model
    init_ema_weights(model, ema_model)

    model.train()
    model.cuda(args.gpu)

    ema_model.train()
    ema_model.cuda(args.gpu)

    cudnn.benchmark = True

    args.snapshot_dir = os.path.join('./data/snapshots', args.file_name)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    cw_loader = data.DataLoader(cityscapesDataset(args.data_dir_cw, args.data_list_cw,
                                                         max_iters=args.num_steps * args.batch_size, mean=IMG_MEAN, set=args.set),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    rf_loader = data.DataLoader(foggyzurichDataSet(args.data_dir_rf, args.data_list_rf,
                                                   max_iters=args.num_steps * args.batch_size, mean=IMG_MEAN, set=args.set),
                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    rf_loader_iter = enumerate(rf_loader)
    cw_loader_iter = enumerate(cw_loader)

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    opts = make_list(optimisers)

    for i_iter in tqdm(range(start_iter, args.num_steps)): 
        loss_seg_cw_value = 0
        loss_seg_mix_value = 0
        loss_seg_rf_value = 0
        if i_iter > 0:
            update_ema_weights(model, ema_model, args.alpha, i_iter)

        for opt in opts:
            opt.zero_grad()
        
        _, batch_cw = cw_loader_iter.__next__()
        image_cw, label_cw, size_cw, name_cw = batch_cw
        interp = nn.Upsample(size=(size_cw[0][0],size_cw[0][1]), mode='bilinear')
        images_cw = Variable(image_cw).cuda(args.gpu)
        
        _, _, _, _, _, feature_cw5 = model(images_cw)
        pred_cw5 = interp(feature_cw5)
        loss_seg_cw = loss_calc(pred_cw5, label_cw, args.gpu)
        
        
        _, batch_rf = rf_loader_iter.__next__()
        image_rf, size_rf, name_rf = batch_rf
        images_rf = Variable(image_rf).cuda(args.gpu)
        
        with torch.no_grad():
            style0, style1, style2, style3, style4, _ = model(images_rf)
            style = (style0, style1, style2, style3, style4)

            _, _, _, _, _, feature_rf5 = ema_model(images_rf)
            ema_pred_rf5 = interp(feature_rf5)
            prob_rf = torch.softmax(ema_pred_rf5.detach(), dim=1)
            conf, pseudo_label = torch.max(prob_rf, dim=1)                   
            energy_rf = -(torch.logsumexp(ema_pred_rf5.detach(), dim=1))
            rf_pseudo_mask = energy_rf.le(args.energy_threshold).long() == 1           
            pseudo_weight = rf_pseudo_mask

        _, _, _, _, _, feature_mix5 = model(images_cw, style)
        pred_mix5 = interp(feature_mix5)
        loss_seg_mix = loss_calc(pred_mix5, label_cw, args.gpu)
        
        _, _, _, _, _, feature_rf5 = model(images_rf)
        pred_rf5 = interp(feature_rf5)
        
        loss_seg_rf = loss_calc_un(pred_rf5, pseudo_label, args.gpu, pseudo_weight)

        loss = (loss_seg_cw + loss_seg_mix) + args.pseudo_weight * loss_seg_rf
        
        loss.backward()

        if loss_seg_cw != 0:
            loss_seg_cw_value += loss_seg_cw.data.cpu().numpy()
        if loss_seg_mix != 0:
            loss_seg_mix_value += loss_seg_mix.data.cpu().numpy()
        if loss_seg_rf != 0:
            loss_seg_rf_value += loss_seg_rf.data.cpu().numpy()


        wandb.log({'CW_loss_seg': loss_seg_cw_value}, step=i_iter)
        wandb.log({'Mix_loss_seg': loss_seg_mix_value}, step=i_iter)
        wandb.log({'RF_loss_seg': loss_seg_rf_value}, step=i_iter)
        wandb.log({'total_loss': loss}, step=i_iter)           

        for opt in opts:
            opt.step()

        if i_iter < 20000:
            save_pred_every = 5000
        else:
            save_pred_every = args.save_pred_every

        if i_iter >= args.num_steps_stop - 1:
            print('save model ..')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.file_name + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save({
                'state_dict':model.state_dict(),
                'ema_state_ditc':ema_model.state_dict(),
                'train_iter':i_iter,
                'args':args
            },osp.join(args.snapshot_dir, run_name)+'-' + str(i_iter)+'.pth')
            

if __name__ == '__main__':
    main()