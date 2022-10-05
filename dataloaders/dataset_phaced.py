import torch
import numpy as np
import cv2
import glob
import json
# from dataloaders.utils_distribution_analysis import visualize_histogram


class PhacedDataset(torch.utils.data.Dataset):
    """Dataloader for our phaced rendered image set.

    Longer class information...
    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
    def __init__(self, folderPath, imageSize, isNormalized, envmapApprox, norm_type):
        folderList = glob.glob(folderPath + '/*')
        imageList = []
        for ifolder in folderList:
            imageList += glob.glob(ifolder + '/*Diffuse.exr')

        self.imageList = imageList
        self.imageSize = imageSize
        self.isNormalized = isNormalized
        self.normalizationType = norm_type
        self.envmapApprox = envmapApprox

        # Measured values, self explanatory
        self.postexposureMean = 0.5
        self.orgEnvmapData = {
                    "exposure_correction" : 886.0557,
                    "mean" : 0.5066,
                    "std" : 0.5237
        }
        self.normalMapData = {
                    "mean_org" : 389.976,
                    "std_org" : 166.197
        }
        self.specularMapData = {
                    "mean_org" : 0.1739,
                    "std_org" : 0.5561
        }
        self.roughnessMapData = {
                    "mean_org" : 168.82478,
                    "std_org" : 207.88754
        }
        self.rhoMapData = {
                    "mean_org" : 170.67735,
                    "std_org" : 213.33794
        }
        self.albedoMapData = {
                    "mean_org" : 95.91259,
                    "std_org" : 136.12364
        }
        self.faceMapData = {
                    "mean_org" : 0.70482045,
                    "std_org" : 1.3049092
        }
        self.diffuseMapData = {
                    "mean_org" : 0.538023,
                    "std_org" : 0.91494614
        }
        self.shadingMapData = {
                    "mean_org" : 2.2303545,
                    "std_org" : 2.9180617
        }

    def __getitem__(self, index):
        imagePath = self.imageList[index]
        mask = cv2.imread(imagePath.replace(
            'Diffuse', 'Mask'), cv2.IMREAD_UNCHANGED)[...,:-1]
        #mask = cv2.GaussianBlur(mask,(5,5),0)
        albedo = cv2.imread(imagePath.replace(
            'Diffuse', 'Albedo'), cv2.IMREAD_UNCHANGED)[...,:-1] * mask
        normal = cv2.imread(imagePath.replace(
            'Diffuse', 'Normal'), cv2.IMREAD_UNCHANGED)[...,:-1]
        normal = ((normal + 1)/2.0)*mask  # temp transform
        roughness = cv2.imread(imagePath.replace(
            'Diffuse', 'Roughness_map'), cv2.IMREAD_UNCHANGED)[...,:-1]
        rho = cv2.imread(imagePath.replace(
            'Diffuse', 'Rho_map'), cv2.IMREAD_UNCHANGED)[...,:-1]
        specular_mask = np.zeros(normal.shape)
        specular_mask[roughness > 0] = 1.0

        if self.isNormalized:
            face = cv2.imread(imagePath.replace(
                'Diffuse', 'Final_Without_SSS_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            diffuse = cv2.imread(imagePath.replace(
                'Diffuse', 'Diffuse_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            specular = cv2.imread(imagePath.replace(
                'Diffuse', 'Specular_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            normal = cv2.imread(imagePath.replace(
                'Diffuse', 'Normal_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            roughness = cv2.imread(imagePath.replace(
                'Diffuse', 'Roughness_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            rho = cv2.imread(imagePath.replace(
                'Diffuse', 'Rho_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            albedo = cv2.imread(imagePath.replace(
                'Diffuse', 'Albedo_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            shading = cv2.imread(imagePath.replace(
                'Diffuse', 'Shading_naivelyscaled'), cv2.IMREAD_UNCHANGED)
            envmap = cv2.imread(imagePath.replace(
                'Diffuse', 'envmap_org_small'), cv2.IMREAD_UNCHANGED)

        else: 
            diffuse = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)[...,:-1]
            face = cv2.imread(imagePath.replace(
                'Diffuse', 'Final_Without_SSS'), cv2.IMREAD_UNCHANGED)[...,:-1]
            specular = face - diffuse
            # We are only dealing with single channel specular, roughness and rho maps
            specular = cv2.cvtColor(specular, cv2.COLOR_BGR2GRAY)
            roughness = cv2.cvtColor(roughness, cv2.COLOR_BGR2GRAY)
            rho = cv2.cvtColor(rho, cv2.COLOR_BGR2GRAY)
            # Get Shading layer. shading = diffuse / albedo
            # shading_mask = (albedo==0.0) & (diffuse==0.0) # Vikas
            shading_mask = (albedo==0.0) # Qian: so no complaints of "RuntimeWarning: invalid value encountered in true_divide"
            shading = np.where(shading_mask, 0, diffuse/(albedo+1e-6))
            shading[shading > 1000] = 0.0
            naiveScale = -1

            if self.normalizationType == "Naive" :
                print("Applying Naive normalization ...")
                # image normalization
                naiveScale = np.max(face)
                print(imagePath, naiveScale)

                if naiveScale > 0:
                    face /= naiveScale
                    diffuse /= naiveScale
                    specular /= naiveScale
                    # normal /= naiveScale

                envmap = self.load_envmap(imagePath, naiveScale)

            elif self.normalizationType == "Weber" :
                # print("Applying Weber et al. normalization...")

                envmap = self.load_envmap(imagePath, -1)

                # Compute envmaps mean
                org_env_median = np.median(envmap)

                # We don't want negative or zero median, thus revert to using mean.
                if org_env_median <= 0:
                    print('Negative or zero envmap median in Weber normalization: ' + imagePath)
                    org_env_median = np.mean(envmap)

                # Scale envmaps
                envmap = envmap * (self.postexposureMean / (org_env_median+1e-6))

                # Exposure correct and Standardize
                if self.envmapApprox == "ORG":
                    # Exposure correction
                    normal = normal * self.orgEnvmapData["exposure_correction"]
                    specular = specular * self.orgEnvmapData["exposure_correction"]
                    roughness = roughness * self.orgEnvmapData["exposure_correction"]
                    rho = rho * self.orgEnvmapData["exposure_correction"]
                    albedo = albedo * self.orgEnvmapData["exposure_correction"]
                    face = face * self.orgEnvmapData["exposure_correction"]
                    diffuse = diffuse * self.orgEnvmapData["exposure_correction"]
                    shading = shading * self.orgEnvmapData["exposure_correction"]
                    # Standardize
                    normal = (normal - self.normalMapData["mean_org"]) / self.normalMapData["std_org"]
                    specular = (specular - self.specularMapData["mean_org"]) / self.specularMapData["std_org"]
                    roughness = (roughness - self.roughnessMapData["mean_org"]) / self.roughnessMapData["std_org"]
                    rho = (rho - self.rhoMapData["mean_org"]) / self.rhoMapData["std_org"]
                    albedo = (albedo - self.albedoMapData["mean_org"]) / self.albedoMapData["std_org"]
                    face = (face - self.faceMapData["mean_org"]) / self.faceMapData["std_org"]
                    diffuse = (diffuse - self.diffuseMapData["mean_org"]) / self.diffuseMapData["std_org"]
                    shading = (shading - self.shadingMapData["mean_org"]) / self.shadingMapData["std_org"]

                # Compute log of envmaps
                envmap = np.log1p(envmap)

                # Standardize
                envmap = (envmap - self.orgEnvmapData["mean"]) / self.orgEnvmapData["std"]

            # self.visualize_histogram(envmap, "hist_org_envmap.png")
            # self.visualize_histogram(normal, "hist_normal.png")
            # self.visualize_histogram(specular, "hist_specular.png")

            # print("Saving normalized inputs...")
            # Saving normalized inputs
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'envmap_org_small'), envmap)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Final_Without_SSS_naivelyscaled'), face)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Diffuse_naivelyscaled'), diffuse)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Specular_naivelyscaled'), specular)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Normal_naivelyscaled'), normal)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Roughness_naivelyscaled'), roughness)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Rho_naivelyscaled'), rho)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Albedo_naivelyscaled'), albedo)
            cv2.imwrite(imagePath.replace(
                'Diffuse', 'Shading_naivelyscaled'), shading)


        face = cv2.resize(face, (self.imageSize, self.imageSize))
        diffuse = cv2.resize(diffuse, (self.imageSize, self.imageSize))
        specular = cv2.resize(specular, (self.imageSize, self.imageSize))
        mask = cv2.resize(mask, (self.imageSize, self.imageSize))
        albedo = cv2.resize(albedo, (self.imageSize, self.imageSize))
        shading = cv2.resize(shading, (self.imageSize, self.imageSize))
        normal = cv2.resize(normal, (self.imageSize, self.imageSize))
        roughness = cv2.resize(roughness, (self.imageSize, self.imageSize))
        rho = cv2.resize(rho, (self.imageSize, self.imageSize))
        specular_mask = cv2.resize(specular_mask, (self.imageSize, self.imageSize))

        # RGB float32
        face = np.asarray(face[..., ::-1].copy())
        face = torch.from_numpy(face).float()  # float32
        diffuse = np.asarray(diffuse[..., ::-1].copy())
        diffuse = torch.from_numpy(diffuse).float()  # float32
        # add single channel dimension to specular
        specular = np.expand_dims(specular, axis=2)
        specular = np.asarray(specular[..., ::-1].copy())
        specular = torch.from_numpy(specular).float()  # float32
        mask = np.asarray(mask[..., ::-1].copy())
        mask = torch.from_numpy(mask).float()  # float32
        albedo = np.asarray(albedo[..., ::-1].copy())
        albedo = torch.from_numpy(albedo).float()  # float32
        shading = np.asarray(shading[..., ::-1].copy())
        shading = torch.from_numpy(shading).float()  # float32
        normal = np.asarray(normal[..., ::-1].copy())
        normal = torch.from_numpy(normal).float()  # float32
        # add single channel dimension to roughness
        roughness = np.expand_dims(roughness, axis=2)
        roughness = np.asarray(roughness[..., ::-1].copy())
        roughness = torch.from_numpy(roughness).float()  # float32
        # add single channel dimension to rho
        rho = np.expand_dims(rho, axis=2)
        rho = np.asarray(rho[..., ::-1].copy())
        rho = torch.from_numpy(rho).float()  # float32
        envmap = np.asarray(envmap[..., ::-1].copy())
        envmap = torch.from_numpy(envmap).float() 
        specular_mask = np.asarray(specular_mask[..., ::-1].copy())
        specular_mask = torch.from_numpy(specular_mask).float()

        return face.permute(2, 0, 1), diffuse.permute(2, 0, 1), specular.permute(2, 0, 1), \
            mask.permute(2, 0, 1), albedo.permute(2, 0, 1), shading.permute(2, 0, 1), normal.permute(2, 0, 1), \
            roughness.permute(2, 0, 1), rho.permute(2, 0, 1), envmap.permute(2, 0, 1), specular_mask.permute(2, 0, 1)

    def __len__(self):
        return len(self.imageList)

    def load_envmap(self, imagePath, naiveScale):
        f = open(imagePath.replace('1_Diffuse.exr', 'info.json'))
        envmap_name = json.load(f)['envmap_name'].split('/')[-1]
        envmap_org = cv2.imread('data/envmaps/'+envmap_name, -1)
        
        envmap = np.fliplr(envmap_org)
        envmap = np.roll(envmap, int(envmap.shape[1]/2), axis=1)
        envmap = envmap[..., ::-1].copy()
        if naiveScale > 0:
            print("Normalizing envmap naively...")
            envmap /= naiveScale  # normalization
        envmap = cv2.resize(cv2.cvtColor(envmap, cv2.COLOR_BGR2RGB), (32,16))

        return envmap

    def undo_normalization(self, *args):
        """
            This function performs the inverse of Weber's normalization.
            
        """

        def expand_channel(data, shape):
            data_3ch = np.zeros(shape)
            data_3ch[:,:,0] = data[:,:,0]
            data_3ch[:,:,1] = data[:,:,0]
            data_3ch[:,:,2] = data[:,:,0]
            return data_3ch
           
        # print("Undo normalization...")

        gt_albedo = args[0].cpu().detach().numpy()
        gt_normal = args[1].cpu().detach().numpy()
        gt_rho = args[2].cpu().detach().numpy()
        gt_roughness = args[3].cpu().detach().numpy()
        gt_shading = args[4].cpu().detach().numpy()
        gt_diffuse = args[5].cpu().detach().numpy()
        gt_specular = args[6].cpu().detach().numpy()
        gt_face = args[7].cpu().detach().numpy()
        envmap = args[8].cpu().detach().numpy()

        pred_albedo = args[9].cpu().detach().numpy()
        pred_normal = args[10].cpu().detach().numpy()
        pred_rho = args[11].cpu().detach().numpy()
        pred_roughness = args[12].cpu().detach().numpy()
        pred_shading = args[13].cpu().detach().numpy()
        pred_diffuse = args[14].cpu().detach().numpy()
        pred_specular = args[15].cpu().detach().numpy()
        pred_recon = args[16].cpu().detach().numpy()
        pred_envmap = args[17].cpu().detach().numpy()
        pred_confidence = args[18].cpu().detach().numpy()

        # Undo torch permute
        gt_albedo = gt_albedo.transpose(1, 2, 0)
        gt_normal = gt_normal.transpose(1, 2, 0)
        gt_rho = gt_rho.transpose(1, 2, 0)
        gt_roughness = gt_roughness.transpose(1, 2, 0)
        gt_shading = gt_shading.transpose(1, 2, 0)
        gt_diffuse = gt_diffuse.transpose(1, 2, 0)
        gt_specular = gt_specular.transpose(1, 2, 0)
        gt_face = gt_face.transpose(1, 2, 0)
        envmap = envmap.transpose(1, 2, 0)

        pred_albedo = pred_albedo.transpose(1, 2, 0)
        pred_normal = pred_normal.transpose(1, 2, 0)
        pred_rho = pred_rho.transpose(1, 2, 0)
        pred_roughness = pred_roughness.transpose(1, 2, 0)
        pred_shading = pred_shading.transpose(1, 2, 0)
        pred_diffuse = pred_diffuse.transpose(1, 2, 0)
        pred_specular = pred_specular.transpose(1, 2, 0)
        pred_recon = pred_recon.transpose(1, 2, 0)
        pred_envmap = pred_envmap.transpose(1, 2, 0)
        pred_confidence = pred_confidence.transpose(1, 2, 0)

        # Undo Standardization for envmaps
        envmap = envmap * self.orgEnvmapData["std"] + self.orgEnvmapData["mean"]
        envmap = np.expm1(envmap)
        org_env_median = np.median(envmap)
        if org_env_median <= 0:
            org_env_median = np.mean(envmap)
        # Scale envmaps
        envmap /= (self.postexposureMean / org_env_median)

        # Exposure correct Normals and Specular and Standardize
        if self.envmapApprox == "ORG":
            exposure_correction = self.orgEnvmapData["exposure_correction"]
            mean = "mean_org"
            std = "std_org"
            envmap_mean = self.orgEnvmapData["mean"]
            envmap_std = self.orgEnvmapData["std"]
            median = org_env_median

        # Undo Standardization for rest
        gt_albedo = gt_albedo * self.albedoMapData[std] + self.albedoMapData[mean]
        gt_albedo /= exposure_correction
        gt_normal = gt_normal * self.normalMapData[std] + self.normalMapData[mean]
        gt_normal /= exposure_correction
        gt_rho = gt_rho * self.rhoMapData[std] + self.rhoMapData[mean]
        gt_rho /= exposure_correction
        gt_roughness = gt_roughness * self.roughnessMapData[std] + self.roughnessMapData[mean]
        gt_roughness /= exposure_correction
        gt_shading = gt_shading * self.shadingMapData[std] + self.shadingMapData[mean]
        gt_shading /= exposure_correction
        gt_diffuse = gt_diffuse * self.diffuseMapData[std] + self.diffuseMapData[mean]
        gt_diffuse /= exposure_correction
        gt_specular = gt_specular * self.specularMapData[std] + self.specularMapData[mean]
        gt_specular /= exposure_correction
        gt_face = gt_face * self.faceMapData[std] + self.faceMapData[mean]
        gt_face /= exposure_correction

        pred_albedo = pred_albedo * self.albedoMapData[std] + self.albedoMapData[mean]
        pred_albedo /= exposure_correction
        pred_normal = pred_normal * self.normalMapData[std] + self.normalMapData[mean]
        pred_normal /= exposure_correction
        pred_rho = pred_rho * self.rhoMapData[std] + self.rhoMapData[mean]
        pred_rho /= exposure_correction
        pred_roughness = pred_roughness * self.roughnessMapData[std] + self.roughnessMapData[mean]
        pred_roughness /= exposure_correction
        pred_shading = pred_shading * self.shadingMapData[std] + self.shadingMapData[mean]
        pred_shading /= exposure_correction
        pred_diffuse = pred_diffuse * self.diffuseMapData[std] + self.diffuseMapData[mean]
        pred_diffuse /= exposure_correction
        pred_specular = pred_specular * self.specularMapData[std] + self.specularMapData[mean]
        pred_specular /= exposure_correction
        pred_recon = pred_recon * self.faceMapData[std] + self.faceMapData[mean]
        pred_recon /= exposure_correction

        pred_envmap = pred_envmap * envmap_std + envmap_mean
        pred_envmap = np.expm1(pred_envmap)
        # confidence from weighted-average
        pred_confidence = pred_confidence * envmap_std + envmap_mean
        pred_confidence = np.expm1(pred_confidence)

        pred_envmap /= (self.postexposureMean / median)
        pred_confidence /= (self.postexposureMean / median)

        gt_specular_3c = expand_channel(gt_specular, gt_normal.shape)
        pred_specular_3c = expand_channel(pred_specular, gt_normal.shape)
        gt_roughness_3c = expand_channel(gt_roughness, gt_normal.shape)
        gt_rho_3c = expand_channel(gt_rho, gt_normal.shape)
        pred_roughness_3c = expand_channel(pred_roughness, gt_normal.shape)
        pred_rho_3c = expand_channel(pred_rho, gt_normal.shape)
        pred_confidence_3c = expand_channel(pred_confidence, [pred_confidence.shape[0], pred_confidence.shape[1], 3])

        return gt_albedo, gt_normal, gt_rho_3c, gt_roughness_3c, gt_shading, gt_diffuse, gt_specular_3c, gt_face, \
            envmap, pred_albedo, pred_normal, pred_rho_3c, pred_roughness_3c, pred_shading, pred_diffuse, \
            pred_specular_3c, pred_recon, pred_envmap, pred_confidence_3c
