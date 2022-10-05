import numpy as np
import cv2
import glob

"""The script is used for analysis only."""

def load_normal(imagePath):
	normal = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
	normal = ((normal + 1)/2.0)
	return normal

def load_specular(imagePath):
	diffuse = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
	face = cv2.imread(imagePath.replace(
	    'Diffuse', 'Final_Without_SSS'), cv2.IMREAD_UNCHANGED)
	specular = face - diffuse
	specular = cv2.cvtColor(specular, cv2.COLOR_BGR2GRAY)
	return specular, face, diffuse

def load_specular_maps(imagePath):
	specular_map = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
	return specular_map

def load_albedo_maps(imagePath):
	diffuse = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
	albedo_map = cv2.imread(imagePath.replace(
	    'Diffuse', 'Albedo'), cv2.IMREAD_UNCHANGED)
	mask = (albedo_map==0.0) & (diffuse==0.0)
	shading = np.where(mask, 0, diffuse/albedo_map)
	shading[shading > 1000] = 0.0
	return albedo_map, shading

def load_envmap(imagePath):
	# calculate sh lighting
	envmap_org = cv2.imread(imagePath, -1)
	envmap = np.fliplr(envmap_org)
	envmap = np.roll(envmap, int(envmap.shape[1]/2), axis=1)
	envmap = envmap[..., ::-1].copy()
	# ch = envmap.shape[2]
	# order = 2
	# envmap_sh = np.zeros((2*order+2, (2*order+2)*2, ch), dtype=np.float32)
	# for i in range(ch):
	#     sh_matrix = shtools_getSH(envmap[:, :, i], order=order)
	#     envmap_sh[..., i] = pyshtools.expand.MakeGridDH(
	# 	sh_matrix, lmax=order, norm=1, sampling=2)
	envmap = cv2.resize(cv2.cvtColor(envmap, cv2.COLOR_BGR2RGB), (32,16))
	# envmap_sh = cv2.resize(envmap_sh, (envmap.shape[1], envmap.shape[0]))
	# envmap_sh = envmap_sh[..., ::-1]
	return envmap

def compute_mean_exposure_correction(envMap):
	# median of each HDRs
	# exposure correction = postexposureMean / median
	# mean of all
	median = np.median(envMap)
	exposureCorrection = 0.5 / median
	return envMap * exposureCorrection, exposureCorrection

def compute_mean_std_dataset(data):
	# for normal, specular and envmaps,
	# compute mean and std for all
	# do this on altered ones
	return np.mean(data), np.std(data)

def compute_envmap(EXRPath):
	EXRs_Mean = []
	EXRs_Std = []
	exposureCorrections = []
	for exr in EXRPath:
		EXR = load_envmap(exr)
		EXRCorrected, exposureCorrection = compute_mean_exposure_correction(EXR)
		EXRCorrected = np.log1p(np.clip(EXRCorrected,0,None))
		mean, std = compute_mean_std_dataset(EXRCorrected)
		EXRs_Mean.append(mean)
		EXRs_Std.append(std)
		exposureCorrections.append(exposureCorrection)
	exposureCorrections = np.asarray(exposureCorrections)
	EXRs_Mean = np.asarray(EXRs_Mean)
	EXRs_Std = np.asarray(EXRs_Std)
	return np.mean(exposureCorrections), np.mean(EXRs_Mean), np.average(EXRs_Std)

# def compute_renders(root, imageType, correction):
# 	means = []
# 	stds = []
# 	for i in range(1, 8700, 2):
# 		imagePath = root + str(i) + imageType
# 		normal = load_specular(imagePath)
# 		normal = normal * correction
# 		mean, std = compute_mean_std_dataset(normal)
# 		means.append(mean)
# 		stds.append(std)
# 	means = np.asarray(means)
# 	stds = np.asarray(stds)
# 	return np.mean(means), np.average(stds)

def compute_facesss(root, imageType, correction):
	face_means = []
	face_stds = []
	diffuse_mean = []
	diffuse_std = []
	for i in range(1, 8700, 2):
		imagePath = root + str(i) + imageType
		specular, face, diffuse = load_specular(imagePath)
		face = face * correction
		mean, std = compute_mean_std_dataset(face)
		face_means.append(mean)
		face_stds.append(std)
		diffuse = diffuse * correction
		mean, std = compute_mean_std_dataset(diffuse)
		diffuse_mean.append(mean)
		diffuse_std.append(std)
		print(i)
	return np.mean(np.asarray(face_means)), np.average(np.asarray(face_stds)), \
		np.mean(np.asarray(diffuse_mean)), np.average(np.asarray(diffuse_std))

def compute_maps(root, imageType, correction):
	albedo_means = []
	albedo_stds = []
	shading_means = []
	shading_stds = []
	for i in range(1, 8700, 5):
		imagePath = root + str(i) + imageType
		albedo, shading = load_albedo_maps(imagePath)
		albedo = albedo * correction
		mean, std = compute_mean_std_dataset(albedo)
		albedo_means.append(mean)
		albedo_stds.append(std)
		shading = shading * correction
		mean, std = compute_mean_std_dataset(shading)
		shading_means.append(mean)
		shading_stds.append(std)
	return np.mean(np.asarray(albedo_means)), np.average(np.asarray(albedo_stds)), \
				np.mean(np.asarray(shading_means)), np.average(np.asarray(shading_stds))

postexposureMean = 0.5
orgEnvmapData = {
			"exposure_correction" : 886.0557,
			"mean" : 0.5066,
			"std" : 0.5237
}
shEnvmapData = {
			"exposure_correction" : 381.2146,
			"mean" : 0.4898,
			"std" : 0.3969
}

face_mean_org, face_std_org, diffuse_mean_org, diffuse_std_org = \
compute_facesss("Aligned-train_facescape-ID_1-100_SH-Envmap/","/1_Diffuse.exr", shEnvmapData["exposure_correction"])

print(face_mean_org, face_std_org, diffuse_mean_org, diffuse_std_org)

albedo_mean_org, albedo_std_org, shading_mean_org, shading_std_org = \
 compute_maps("Aligned-train_facescape-ID_1-100_SH-Envmap/","/1_Diffuse.exr", shEnvmapData["exposure_correction"])

print(albedo_mean_org, albedo_std_org, shading_mean_org, shading_std_org)

# >>> print(face_mean_org, face_std_org, diffuse_mean_org, diffuse_std_org)
# 0.36669078 0.60561705 0.2829114 0.47713497

# >>> print(albedo_mean_org, albedo_std_org, shading_mean_org, shading_std_org)
# 41.265213 58.565525 -inf nan