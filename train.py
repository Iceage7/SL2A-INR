
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from tqdm.notebook import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_msssim import ssim

from modules import utils
from modules.models import INR
from SL2A_INR import SL2A
from Finer import Finer


parser = argparse.ArgumentParser(description='Image')

# Shared Parameters
parser.add_argument('--input',type=str, default='./data/00.png', help='Input image path')
parser.add_argument('--inr_model',type=str, default='sl2a', help='(gauss, relu, siren, wire, finer, sl2a)')
parser.add_argument('--using_schedular', type=bool, default=True, help='Whether to use schedular')
parser.add_argument('--scheduler_b', type=float, default=0.1, help='Learning rate scheduler')
parser.add_argument('--niters', type=int, default=501, help='Number if iterations')
parser.add_argument('--steps_til_summary', type=int, default=1001, help='Number of steps till summary visualization')
parser.add_argument('--deg', type=int, default=256, help="degree of sl2a")
parser.add_argument('--hidden_layers', type=int, default=3, help="number of hidden layers")
parser.add_argument('--hidden_features', type=int, default=256, help='width of each hidden layer')
parser.add_argument('--in_features', type=int, default=2, help="number of input features")
parser.add_argument('--out_features', type=int, default=3, help="number of output features")
parser.add_argument('--factor', type=int, default=4, help="down scale factor")
parser.add_argument('--first_omega_0', type=int, default=30, help="(siren, wire, finer)")
parser.add_argument('--hidden_omega_0', type=float, default=30, help='(siren, wire, finer)')    
parser.add_argument('--scale', type=float, default=30, help='simga (wire, guass)') 
parser.add_argument('--rank', type=int, default=128, help='rank (sl2a)')
parser.add_argument('--nonlinearity', type=str, default='relu', help='(relu, None) (sl2a)') 
parser.add_argument('--init_method', type=str, default='xavier_uniform', help='(xavier_uniform, kaiming_normal, kaiming_uniform, orthogonal, uniform, normal) (sl2a)') 
parser.add_argument('--linear_init_type', type=str, default='kaiming_uniform', help='(xavier_uniform, kaiming_normal, kaiming_uniform, orthogonal, uniform, normal) (sl2a)') 
parser.add_argument('--sl2a_lr', type=float, default=0.001, help='learning rate (sl2a)')
parser.add_argument('--sl2a_maxpoints', type=int, default=16384, help='maxpoints (sl2a)')

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    factor = args.factor
    im = utils.normalize(plt.imread(args.input).astype(np.float32), True)
    im = cv2.resize(im, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape
    print(f'Image shape: {im.shape}')
    seed = 42 
    set_seed(seed)
    pos_encode_freq = {'type':'frequency', 'use_nyquist': True, 'mapping_input': int(max(H, W)/3)}

    pos_encode_no = {'type': None}

    model_type = args.inr_model


    if model_type == 'sl2a':

        model = SL2A(in_features = args.in_features, out_features=args.out_features,
                            hidden_layers = args.hidden_layers,
                            hidden_features = args.hidden_features, deg=args.deg, rank=args.rank, nonlinearity=args.nonlinearity, init_method=args.init_method,
                            linear_init_type=args.linear_init_type
                            ).to(device)

        args.lr = args.sl2a_lr
        args.maxpoints = args.sl2a_maxpoints


    elif model_type == 'siren':
        model = INR('siren').run(
            in_features=args.in_features,
            out_features=args.out_features,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
            first_omega_0=args.first_omega_0,
            hidden_omega_0=args.hidden_omega_0,
            pos_encode_configs=pos_encode_no).to(device)

        args.lr = 0.001
        args.maxpoints = 65536

    elif model_type== 'gauss':
        model = INR('gauss').run(
            in_features=args.in_features,
            out_features=args.out_features,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
            sigma=args.scale,
            pos_encode_configs=pos_encode_no).to(device)
        
        args.lr = 0.0001
        args.maxpoints = 1024

    elif model_type == 'relu':
        model = INR('relu').run(
            in_features=args.in_features,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
            out_features=args.out_features,
            pos_encode_configs=pos_encode_freq
        ).to(device)


        args.lr = 0.001
        args.maxpoints = 1024

    elif model_type == "wire":


        model = INR('wire').run(
            in_features=args.in_features,
            out_features=args.out_features,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
            first_omega_0=args.first_omega_0,
            hidden_omega_0=args.hidden_omega_0,
            sigma=args.scale,
            wire_type='complex',
            pos_encode_configs=pos_encode_no).to(device)
        
        args.lr = 0.001
        args.maxpoints = 65536


    elif model_type == 'finer':
        model = Finer(in_features=args.in_features, out_features=args.out_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features,
                first_omega_0=args.first_omega_0, hidden_omega_0=args.hidden_omega_0, first_bias_scale=20, scale_req_grad=False).to(device)

        args.lr = 0.0001
        args.maxpoints = 16384

    print(f'Total number of training parameters: {count_parameters(model)}')


    optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = lr_scheduler.LambdaLR(optim, lambda x: args.scheduler_b ** min(x / args.niters, 1))


    psnr_values = []
    ms_ssim_values = []
    mse_array = torch.zeros(args.niters, device=device)
    best_loss = torch.tensor(float('inf'))
    coords = utils.get_coords(H, W, dim=2)[None, ...]
    gt = torch.tensor(im).reshape(H * W, 3)[None, ...].to(device)
    rec = torch.zeros_like(gt)
    for step in tqdm(range(args.niters)):

        indices = torch.randperm(H*W)
        for b_idx in range(0, H*W, args.maxpoints):
            b_indices = indices[b_idx:min(H*W, b_idx+args.maxpoints)]
            b_coords = coords[:, b_indices, ...].to(device)
            b_indices = b_indices.to(device)
            
            model_output = model(b_coords)
            with torch.no_grad():
                rec[:, b_indices, :] = model_output

            loss = ((model_output - gt[:, b_indices, :])**2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            mse_array[step] = ((gt - rec)**2).mean().item()
            psnr = -10*torch.log10(mse_array[step])
            psnr_values.append(psnr.item())


        if args.using_schedular:

            scheduler.step()

        imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()

        if (mse_array[step] < best_loss) or (step == 0):
            
            best_loss = mse_array[step]
            best_img = imrec
            best_img = (best_img - best_img.min()) / (best_img.max() - best_img.min())
            best_model = model


        ms_ssim_val = ssim(torch.tensor(im[None,...]).permute(0, 3, 1, 2),
                                torch.tensor(best_img[None, ...]).permute(0, 3, 1, 2),
                                data_range=1, size_average=False)

        ms_ssim_values.append(ms_ssim_val[0].item())

        if step % args.steps_til_summary == 0:
            print("Epoch: {} | Total Loss: {:.5f} | PSNR: {:.4f} | SSIM: {:.4f}".format(step,
                                                                        mse_array[step].item(),
                                                                    psnr.item(), ms_ssim_val[0].item()))

    with torch.no_grad():

        coords = utils.get_coords(H, W, dim=2)[None, ...].to(device)

        model_output = best_model(coords)

        rec[:, :, :] = model_output

        mse_array[step] = ((gt - rec)**2).mean().item()
        best_psnr = -10*torch.log10(mse_array[step])

        imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()
        best_img = imrec
        best_img = (best_img - best_img.min()) / (best_img.max() - best_img.min())

        ms_ssim_val = ssim(torch.tensor(im[None,...]).permute(0, 3, 1, 2),
                                    torch.tensor(best_img[None, ...]).permute(0, 3, 1, 2),
                                    data_range=1, size_average=False)

        best_ms_ssim = ms_ssim_val[0].item()


    print('--------------------')
    print('PSNR:', best_psnr.item())
    print('SSIM:', best_ms_ssim)
    print('--------------------')


    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    input_name = args.input.split('/')[-1].split('.')[0]
    save_path = os.path.join(save_dir, f'{model_type}_{input_name}.pth')
    torch.save(best_model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")


