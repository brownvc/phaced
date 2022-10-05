import os
import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np
import cv2
from soravux_envmap_tools import EnvironmentMap

from dataloaders.dataset_phaced import PhacedDataset
from models.full_model import UNetModel, DecompModel

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def test(config):
    # init
    torch.autograd.set_detect_anomaly(True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.deterministic = True  # ???
    rd_seed = 1
    random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    torch.cuda.manual_seed(rd_seed)
    np.random.seed(rd_seed)

    if config.stage == "Delight" or config.stage == "Full":
        # De Lighitng network + Diffuse I2I + Specular I2I
        delight_net = DecompModel(config.cond_envmap).cuda()
        # Needs to be replace with partial pre-trained model
        delight_net.load_state_dict(torch.load(config.delight_pretrain_dir))
        delight_net.eval()

    # Diffuse I2I or full model
    if config.stage == "DiffuseI2I" or config.stage == "Full":
        # Diffuse I2I Model
        diffuse_i2i_net = UNetModel(in_channels=3, out_channels=3, cond_envmap=config.cond_envmap).cuda()
        # Needs to be replace with partial pre-trained model
        diffuse_i2i_net.load_state_dict(torch.load(config.diffusei2i_pretrain_dir))
        diffuse_i2i_net.eval()

    # Specular I2I or full model
    if config.stage == "SpecularI2I" or config.stage == "Full":
        # Specular I2I Model
        specular_i2i_net = UNetModel(in_channels=5, cond_envmap=config.cond_envmap).cuda()
        # Needs to be replace with partial pre-trained model
        specular_i2i_net.load_state_dict(torch.load(config.speculari2i_pretrain_dir))
        specular_i2i_net.eval()

    # dataloader
    imageSize = 128
    test_dataset = PhacedDataset(
        config.image_path, imageSize, config.image_isnormalized, config.envmap_approx, config.norm_type)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.test_batch_size, shuffle=False,
                                              num_workers=config.num_workers,
                                              pin_memory=True)  # load your samples in the Dataset on CPU and push it
    # during testing to the GPU to speed up the host to device transfer

    # loss functions
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    # Solid angle for 16x32 envmap
    logloss_solidangle = EnvironmentMap(16, 'LatLong').solidAngles()
    logloss_weight = torch.autograd.Variable(torch.from_numpy(logloss_solidangle), requires_grad=False).float().cuda()

    # loss values
    num_batches = len(test_loader)
    # loss_specular = np.zeros((num_batches, 1))
    # loss = np.zeros((num_batches, 1))


    with torch.no_grad():  # deactivate the autograd engine
        for i, data in enumerate(test_loader):
            # unpack inputs
            face, gt_diffuse, gt_specular, mask, gt_albedo, gt_shading, gt_normal, \
            gt_light, gt_roughness, gt_rho, gt_envmap_org, gt_envmap_sh, gt_specular_mask = data

            face, gt_diffuse, gt_specular, mask, gt_albedo, gt_shading, gt_normal, gt_light, \
            gt_roughness, gt_rho, gt_envmap_org, gt_envmap_sh, gt_specular_mask = face.cuda(), gt_diffuse.cuda(), \
            gt_specular.cuda(), mask.cuda(), gt_albedo.cuda(), gt_shading.cuda(), gt_normal.cuda(), gt_light.cuda(), \
            gt_roughness.cuda(), gt_rho.cuda(), gt_envmap_org.cuda(), gt_envmap_sh.cuda(), gt_specular_mask.cuda()

            # Initialize with ground truth values
            albedo, normal, roughness, rho, confidence, recon, shading, diffuse, specular = gt_albedo,\
             gt_normal, gt_roughness, gt_rho, gt_envmap_org, face, gt_shading, gt_diffuse, gt_specular

            if config.cond_envmap == "ORG":
                envmap = gt_envmap_org
            elif config.cond_envmap == "SH":
                envmap = gt_envmap_sh
            elif config.cond_envmap == "NONE":
                envmap = torch.empty(0)

            # delight or full model
            if config.stage == "Delight" or config.stage == "Full":
                albedo, normal, roughness, rho, envmap, confidence = delight_net(face, mask)

            # Diffuse I2I or full model
            if config.stage == "DiffuseI2I" or config.stage == "Full":
                diffuse_i2i_input = normal
                shading = diffuse_i2i_net(diffuse_i2i_input, envmap)
                diffuse = shading * albedo

            # Specular I2I or full model
            if config.stage == "SpecularI2I" or config.stage == "Full":
                specular_i2i_input = torch.cat((normal, roughness, rho), 1)
                specular = specular_i2i_net(specular_i2i_input, envmap)

            if config.stage == "Full":
                recon = diffuse + specular

            # supervised loss
            # loss_specular[i] = L1(specular, gt_specular).cpu().numpy()

            # NOTE: weight tuning
            # loss_specular_weight = 10
            # loss[i] = loss_specular_weight * loss_specular[i]

            # save images
            # if (i % config.display_iter) == 0:
            for j in range(gt_normal.shape[0]):
                # Undo normalization before writing to images (only doing for one batch)
                gt_albedo_bnp, gt_normal_bnp, gt_rho_bnp, gt_roughness_bnp,                                         \
                gt_shading_bnp, gt_diffuse_bnp, gt_specular_bnp, gt_face_bnp, gt_envmap_org_bnp, gt_envmap_sh_bnp,  \
                albedo_bnp, normal_bnp, rho_bnp, roughness_bnp, shading_bnp, diffuse_bnp, specular_bnp, recon_bnp,  \
                envmap_bnp, confidence_bnp = test_dataset.undo_normalization( gt_albedo[j], gt_normal[j],          \
                gt_rho[j], gt_roughness[j], gt_shading[j], gt_diffuse[j], gt_specular[j], face[j],               \
                gt_envmap_org[j], gt_envmap_sh[j], albedo[j], normal[j], rho[j], roughness[j], shading[j],          \
                diffuse[j], specular[j], recon[j], envmap[j], confidence[j])

                gt_envmap_org_bnp = cv2.resize(gt_envmap_org_bnp, 
                    ((gt_normal_bnp.shape[0] * gt_envmap_org_bnp.shape[1]) // gt_envmap_org_bnp.shape[0], gt_normal_bnp.shape[0]))

                gt_envmap_sh_bnp = cv2.resize(gt_envmap_sh_bnp, 
                    ((gt_normal_bnp.shape[0] * gt_envmap_sh_bnp.shape[1]) // gt_envmap_sh_bnp.shape[0], gt_normal_bnp.shape[0]))

                envmap_bnp = cv2.resize(envmap_bnp, 
                    ((gt_normal_bnp.shape[0] * envmap_bnp.shape[1]) // envmap_bnp.shape[0], gt_normal_bnp.shape[0]))

                confidence_bnp = cv2.resize(confidence_bnp, 
                    ((gt_normal_bnp.shape[0] * confidence_bnp.shape[1]) // confidence_bnp.shape[0], gt_normal_bnp.shape[0]))

                gt_albedo_bnp = cv2.cvtColor(gt_albedo_bnp, cv2.COLOR_BGR2RGB)
                albedo_bnp = cv2.cvtColor(albedo_bnp, cv2.COLOR_BGR2RGB)
                gt_normal_bnp = cv2.cvtColor(gt_normal_bnp, cv2.COLOR_BGR2RGB)
                normal_bnp = cv2.cvtColor(normal_bnp, cv2.COLOR_BGR2RGB)
                gt_shading_bnp = cv2.cvtColor(gt_shading_bnp, cv2.COLOR_BGR2RGB) 
                shading_bnp = cv2.cvtColor(shading_bnp, cv2.COLOR_BGR2RGB) 
                gt_diffuse_bnp = cv2.cvtColor(gt_diffuse_bnp, cv2.COLOR_BGR2RGB) 
                diffuse_bnp = cv2.cvtColor(diffuse_bnp, cv2.COLOR_BGR2RGB) 
                gt_face_bnp =  cv2.cvtColor(gt_face_bnp, cv2.COLOR_BGR2RGB) 
                recon_bnp =  cv2.cvtColor(recon_bnp, cv2.COLOR_BGR2RGB) 

                gt_envmap_org_bnp = cv2.cvtColor(gt_envmap_org_bnp, cv2.COLOR_BGR2RGB) 
                gt_envmap_sh_bnp = cv2.cvtColor(gt_envmap_sh_bnp, cv2.COLOR_BGR2RGB) 
                envmap_bnp = cv2.cvtColor(envmap_bnp, cv2.COLOR_BGR2RGB) 


               # delight or full model
                if config.stage == "Delight" or config.stage == "Full":
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_normal.exr' % (i,j),
                     np.concatenate((gt_normal_bnp.astype('float32'), normal_bnp.astype('float32')), axis=1))
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_albedo.exr' % (i,j),
                     np.concatenate((gt_albedo_bnp.astype('float32'), albedo_bnp.astype('float32')), axis=1))
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_rho_roughness.exr' % (i,j),
                     np.concatenate((gt_rho_bnp.astype('float32'), rho_bnp.astype('float32'), \
                        gt_roughness_bnp.astype('float32'), roughness_bnp.astype('float32')), axis=1))
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_envmap.exr' % (i,j),
                     np.concatenate((gt_envmap_org_bnp.astype('float32'), envmap_bnp.astype('float32')), axis=1))
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_face.exr' % (i,j),
                        gt_face_bnp.astype('float32'))

                # Diffuse I2I or full model
                if config.stage == "DiffuseI2I" or config.stage == "Full":
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_diffuse.exr' % (i,j),
                     np.concatenate((gt_diffuse_bnp.astype('float32'), diffuse_bnp.astype('float32')), axis=1))
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_shading.exr' % (i,j),
                     np.concatenate((gt_shading_bnp.astype('float32'), shading_bnp.astype('float32')), axis=1))

                # Specular I2I or full model
                if config.stage == "SpecularI2I" or config.stage == "Full":
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_specular.exr' % (i,j),
                     np.concatenate((gt_specular_bnp.astype('float32'), specular_bnp.astype('float32')), axis=1))

                if config.stage == "Full":
                    cv2.imwrite(config.results_folder+'/batch%d_image%d_face.exr' % (i,j),
                     np.concatenate((gt_face_bnp.astype('float32'), recon_bnp.astype('float32')), axis=1))

                # print("Weighted Loss : ", loss[i].item(), " , Specular L1 : ", loss_specular[i].item())
            # print("Mean batch loss : ", np.mean(loss[i]))
    # print losses
    print('%d test images' % (num_batches*config.test_batch_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    # path = 'data/facessss_1026_matched45'
    path = ''
    modelid = ''
    parser.add_argument('--image_path', type=str, default=path)
    parser.add_argument('--image_isnormalized', action='store_true')
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--results_folder', type=str,
                        default="results/%s/" % modelid)
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

    if not os.path.exists(config.results_folder):
        os.makedirs(config.results_folder)

    test(config)

""" 
python test_phaced_models.py \
--image_path data/test_synthetic_hdr/ \
--results_folder results/test_synthetic_hdr \
--delight_pretrain_dir pretrained_weights/Delight/phaced_delight-800.pth \
--speculari2i_pretrain_dir pretrained_weights/SpecularI2I/phaced_speculari2i-800.pth \
--diffusei2i_pretrain_dir pretrained_weights/DiffuseI2I/phaced_diffusei2i-800.pth \
--cond_envmap ORG \
--norm_type Weber \
--envmap_approx ORG \
--test_batch_size 8 \
--stage Full
 """