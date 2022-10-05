import os
import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from soravux_envmap_tools import EnvironmentMap

from dataloaders.dataset_phaced import PhacedDataset
from models.full_model import UNetModel, DecompModel


def weights_init(m):
    # NOTE: xavier seems to converge faster
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    if hasattr(m, 'weight') and classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    # BatchNorm?


def train(config):
    # init
    torch.autograd.set_detect_anomaly(True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.deterministic = True
    rd_seed = 1
    random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    torch.cuda.manual_seed(rd_seed)
    np.random.seed(rd_seed)
    # List of trainable parameters
    params = []
    # List of network branches
    network_branches = {"Delight":[],"SpecularI2I":[],"DiffuseI2I":[]}

    if config.stage == "Delight" or config.stage == "Full":
        # De Lighitng network + Diffuse I2I + Specular I2I
        delight_net = DecompModel(config.cond_envmap).cuda()

        delight_net.apply(weights_init)
        # Needs to be replace with partial pre-trained model
        if config.load_pretrain:
            delight_net.load_state_dict(torch.load(config.delight_pretrain_dir))
        # Network parameters
        params += list(delight_net.parameters())
        network_branches["Delight"] = delight_net

    # Diffuse I2I or full model
    if config.stage == "DiffuseI2I" or config.stage == "Full":
        # Diffuse I2I Model
        diffuse_i2i_net = UNetModel(in_channels=3, out_channels=3, cond_envmap=config.cond_envmap).cuda()

        diffuse_i2i_net.apply(weights_init)
        # Needs to be replace with partial pre-trained model
        if config.load_pretrain:
            diffuse_i2i_net.load_state_dict(torch.load(config.diffusei2i_pretrain_dir))
        # Network parameters
        params += list(diffuse_i2i_net.parameters())
        network_branches["DiffuseI2I"] = diffuse_i2i_net

    # Specular I2I or full model
    if config.stage == "SpecularI2I" or config.stage == "Full":
        # Specular I2I Model
        specular_i2i_net = UNetModel(in_channels=3, cond_envmap=config.cond_envmap).cuda()

        specular_i2i_net.apply(weights_init)
        # Needs to be replace with partial pre-trained model
        if config.load_pretrain:
            specular_i2i_net.load_state_dict(torch.load(config.speculari2i_pretrain_dir))
        # Network parameters
        params += list(specular_i2i_net.parameters())
        network_branches["SpecularI2I"] = specular_i2i_net

    # dataloader
    imageSize = 128
    train_dataset = PhacedDataset(
        config.image_path, imageSize, config.image_isnormalized, config.envmap_approx, config.norm_type)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers,
                                               pin_memory=True)  # load your samples in the Dataset on CPU and push it
    # during training to the GPU to speed up the host to device transfer

    # loss
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    # Solid angle for 16x32 envmap
    logloss_solidangle = EnvironmentMap(16, 'LatLong').solidAngles()
    logloss_weight = torch.autograd.Variable(torch.from_numpy(logloss_solidangle), requires_grad=False).float().cuda()

    # optimizer
    optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)

    # approx sfsnet multi-step at iters .....
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 200, 350], gamma=0.1, last_epoch=-1)
     #StepLR(optimizer, step_size=50, gamma=0.1)

    # visualizer
    # $ tensorboard --logdir runs/
    timestr = config.stage + datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(config.runs_folder+timestr)  # default
    if not os.path.exists(config.snapshots_folder+timestr):
        os.makedirs(config.snapshots_folder+timestr)

    for branch in network_branches.values():
        if branch != []:
            branch.train()

    # delight_net.train()
    # specular_i2i_net.train()
    # diffuse_i2i_net.train()

    for epoch in range(config.num_epochs):
        print("------------------Epoch %d / %d--------------------" % (epoch, config.num_epochs))
        for iteration, data in enumerate(train_loader):
            # unpack inputs
            face, gt_diffuse, gt_specular, mask, gt_albedo, gt_shading, gt_normal, \
            gt_roughness, gt_rho, gt_envmap_org, gt_specular_mask = data

            face, gt_diffuse, gt_specular, mask, gt_albedo, gt_shading, gt_normal, \
            gt_roughness, gt_rho, gt_envmap_org, gt_specular_mask = face.cuda(), gt_diffuse.cuda(), \
            gt_specular.cuda(), mask.cuda(), gt_albedo.cuda(), gt_shading.cuda(), gt_normal.cuda(), \
            gt_roughness.cuda(), gt_rho.cuda(), gt_envmap_org.cuda(), gt_specular_mask.cuda()

            # Initialize with ground truth values
            albedo, normal, roughness, rho, confidence, recon, shading, diffuse, specular = gt_albedo,\
             gt_normal, gt_roughness, gt_rho, gt_envmap_org, face, gt_shading, gt_diffuse, gt_specular

            if config.cond_envmap == "ORG":
                envmap = gt_envmap_org
            elif config.cond_envmap == "NONE":
                envmap = torch.empty(0)
            # Set losses to zero
            loss_albedo = loss_normal = loss_rho = loss_roughness = loss_shading = loss_diffuse = \
            loss_specular = loss_recon = loss_envmap = torch.tensor([0.0]).cuda()

            # delight or full model
            if config.stage == "Delight" or config.stage == "Full":
                albedo, normal, roughness, rho, envmap, confidence = delight_net(face, mask)

                # supervised loss
                loss_albedo = L1(albedo, gt_albedo)
                loss_normal = L1(normal, gt_normal)
                loss_rho = L1(rho, gt_rho)
                loss_roughness = L1(roughness, gt_roughness)
                # weighted L1
                if config.cond_envmap == "ORG":
                    loss_envmap = torch.sum(logloss_weight * torch.abs(envmap - gt_envmap_org))

            # Diffuse I2I or full model
            if config.stage == "DiffuseI2I" or config.stage == "Full":
                diffuse_i2i_input = normal
                shading = diffuse_i2i_net(diffuse_i2i_input, envmap)
                diffuse = shading * albedo
                loss_shading = L1(shading, gt_shading)
                loss_diffuse = L1(diffuse, gt_diffuse)

            # Specular I2I or full model
            if config.stage == "SpecularI2I" or config.stage == "Full":
                specular_i2i_input = torch.cat((normal, roughness, rho), 1)
                specular = specular_i2i_net(normal, envmap)
                loss_specular = L1(specular, gt_specular)

            if config.stage == "Full":
                recon = diffuse + specular
                loss_recon = L1(recon, face)

            # NOTE: weight tuning
            loss =  1.0 * (loss_rho + loss_roughness) + \
                    0.8 * (loss_normal + loss_albedo + loss_shading + loss_diffuse + loss_specular + loss_recon) + \
                    0.1 * loss_envmap

            # autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            istep = epoch * len(train_loader) + iteration

            if (iteration % config.display_iter) == 0:
                
                print("Total Loss = %.5f, Loss_albedo = %.5f, Loss_normal = %.5f, loss_rho = %.5f,                 \
                    Loss at loss_roughness = %.5f, loss_shading = %.5f, loss_diffuse = %.5f, loss_specular = %.5f, \
                    Loss at loss_recon = %.5f, loss_envmap = %.5f"
                    % (loss.item(), loss_albedo.item(), loss_normal.item(), loss_rho.item(), loss_roughness.item(),
                        loss_shading.item(), loss_diffuse.item(), loss_specular.item(), loss_recon.item(), loss_envmap.item()))

                writer.add_scalar('loss', loss, istep)

            if (iteration % config.saveresults_iter) == 0 and (epoch % 50) == 0:

                if config.norm_type == 'Weber':
                    # Undo normalization before writing to images (only doing for one batch)
                    gt_albedo_bnp, gt_normal_bnp, gt_rho_bnp, gt_roughness_bnp,                                         \
                    gt_shading_bnp, gt_diffuse_bnp, gt_specular_bnp, gt_face_bnp, gt_envmap_org_bnp,  \
                    albedo_bnp, normal_bnp, rho_bnp, roughness_bnp, shading_bnp, diffuse_bnp, specular_bnp, recon_bnp,  \
                    envmap_bnp, confidence_bnp = train_dataset.undo_normalization( gt_albedo[0], gt_normal[0],          \
                    gt_rho[0], gt_roughness[0], gt_shading[0], gt_diffuse[0], gt_specular[0], face[0],               \
                    gt_envmap_org[0], albedo[0], normal[0], rho[0], roughness[0], shading[0],          \
                    diffuse[0], specular[0], recon[0], envmap[0], confidence[0])

                    gt_envmap_org_bnp = cv2.resize(gt_envmap_org_bnp, 
                        ((gt_normal_bnp.shape[0] * gt_envmap_org_bnp.shape[1]) // gt_envmap_org_bnp.shape[0], gt_normal_bnp.shape[0]))

                    envmap_bnp = cv2.resize(envmap_bnp, 
                        ((gt_normal_bnp.shape[0] * envmap_bnp.shape[1]) // envmap_bnp.shape[0], gt_normal_bnp.shape[0]))

                    confidence_bnp = cv2.resize(confidence_bnp, 
                        ((gt_normal_bnp.shape[0] * confidence_bnp.shape[1]) // confidence_bnp.shape[0], gt_normal_bnp.shape[0]))

                    # Additonal Scale

                    gt_albedo_bnp = cv2.cvtColor(gt_albedo_bnp, cv2.COLOR_BGR2RGB) * 255
                    albedo_bnp = cv2.cvtColor(albedo_bnp, cv2.COLOR_BGR2RGB) * 255
                    gt_normal_bnp = cv2.cvtColor(gt_normal_bnp, cv2.COLOR_BGR2RGB) * 255
                    normal_bnp = cv2.cvtColor(normal_bnp, cv2.COLOR_BGR2RGB) * 255
                    gt_roughness_bnp = gt_roughness_bnp * 255
                    roughness_bnp = roughness_bnp * 255
                    gt_rho_bnp = gt_rho_bnp * 255
                    rho_bnp = rho_bnp * 255
                    gt_shading_bnp = cv2.cvtColor(gt_shading_bnp, cv2.COLOR_BGR2RGB) * 20000
                    shading_bnp = cv2.cvtColor(shading_bnp, cv2.COLOR_BGR2RGB) * 20000
                    gt_diffuse_bnp = cv2.cvtColor(gt_diffuse_bnp, cv2.COLOR_BGR2RGB) * 90000
                    diffuse_bnp = cv2.cvtColor(diffuse_bnp, cv2.COLOR_BGR2RGB) * 90000
                    gt_specular_bnp =  gt_specular_bnp * 90000
                    specular_bnp =  specular_bnp * 90000
                    gt_face_bnp =  cv2.cvtColor(gt_face_bnp, cv2.COLOR_BGR2RGB) * 90000
                    recon_bnp =  cv2.cvtColor(recon_bnp, cv2.COLOR_BGR2RGB) * 90000

                    gt_envmap_org_bnp = cv2.cvtColor(gt_envmap_org_bnp, cv2.COLOR_BGR2RGB) * 100
                    envmap_bnp = cv2.cvtColor(envmap_bnp, cv2.COLOR_BGR2RGB) * 100
                    confidence_bnp = confidence_bnp * 100

                    cv2.imwrite(config.runs_folder+timestr+'/images_epoch%d_batch%d.png' % (epoch, istep),
                        np.concatenate(
                            (np.concatenate((gt_face_bnp, gt_albedo_bnp, gt_normal_bnp, gt_rho_bnp, gt_roughness_bnp, \
                            gt_shading_bnp, gt_diffuse_bnp, gt_specular_bnp, gt_envmap_org_bnp), axis=1),
                            np.concatenate((recon_bnp, albedo_bnp, normal_bnp, rho_bnp, roughness_bnp, \
                            shading_bnp, diffuse_bnp, specular_bnp, envmap_bnp), axis=1)),
                        axis=0))

            if epoch > 100 and (epoch % config.snapshot_iter) == 0 and iteration == len(train_loader)-1:
                for name, branch in zip(network_branches.keys(), network_branches.values()):
                    if branch != []:
                        torch.save(branch.state_dict(
                        ), config.snapshots_folder + timestr + '/facessss_full-model-v2_' + name + '-' + str(epoch) + '.pth')
        
        # serWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you 
        # should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this 
        # will result in PyTorch skipping the first value of the learning rate schedule. 
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()
        print('multi-step lr: %.6f' % (scheduler.get_last_lr()[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    path = ""
    parser.add_argument('--image_path', type=str, default=path)
    parser.add_argument('--image_isnormalized', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--grad_clip_norm', type=float,
                        default=0.1)  # not sure if we need it
    parser.add_argument('--num_epochs', type=int, default=2500)
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--saveresults_iter', type=int, default=50)
    parser.add_argument('--snapshot_iter', type=int, default=250)
    parser.add_argument('--snapshots_folder', type=str,
                        default="snapshots/")
    parser.add_argument('--runs_folder', type=str, default="runs/")
    parser.add_argument('--load_pretrain', action='store_true')
    parser.add_argument('--delight_pretrain_dir', type=str, default='')
    parser.add_argument('--speculari2i_pretrain_dir', type=str, default='')
    parser.add_argument('--diffusei2i_pretrain_dir', type=str, default='')
    parser.add_argument('--cond_envmap', type=str, 
        help='Type of environment map conditioning. Valid values = NONE | ORG | SH', default='ORG')
    parser.add_argument('--norm_type', type=str, 
        help='Type of normalization. Valid values = Weber | Naive', default='Weber')
    parser.add_argument('--envmap_approx', type=str, 
        help='Type of dataset environment map approximation. Valid values = ORG | SH', default='ORG')
    parser.add_argument('--stage', type=str, 
        help='Which stage to train. Branch name: Delight | DiffuseI2I | SpecularI2I | Full', default='Delight')

    config = parser.parse_args()

    if not os.path.exists(config.runs_folder):
        os.mkdir(config.runs_folder)

    if not config.image_isnormalized:
        config.num_epochs = 1

    starttime = datetime.now()
    train(config)
    endtime = datetime.now()
    print('Start time: %s' % starttime.strftime('%Y%m%d-%H:%M:%S'))
    print('End time: %s' % endtime.strftime('%Y%m%d-%H:%M:%S'))
    runtime = endtime - starttime
    print(runtime)
    print('Runtime: %s' % str(runtime)) # ('%H:%M:%S')
    

# train Delight, DiffuseI2I, SpecularI2I branches for 800 epochs, respectively
# then train the model end-to-end in Full mode for 800 epochs
"""
python train_phaced_models.py \
--image_path data/Aligned-train_facescape-ID_1-100_Org-Envmap/ \
--image_isnormalized \
--lr 0.0001 \
--cond_envmap ORG \
--norm_type Weber \
--envmap_approx ORG \
--train_batch_size 8 \
--snapshot_iter 200 \
--num_epochs 800 \
--stage Full
"""