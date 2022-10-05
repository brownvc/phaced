from parse_eth import load_eth_model, initialize_parameters
import argparse
import numpy as np
import time

# https://github.com/opencv/opencv/issues/21326
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


def compute_mean_parameters(m, rs, specular_hyperparam):
	specular_param = []
	for face_region in range(0,9):
		# if using custom colors
		# index = np.array(face_seg) == np.array(segments_color[face_region])

		avg_rs = np.mean(rs[face_region])
		avg_m = np.mean(m[face_region])
		specular_param.append([avg_rs + specular_hyperparam[0], avg_m + specular_hyperparam[1]])
	specular_param = np.array(specular_param)
	return specular_param

def save_map(indv_specular_param, face_seg, face_id):
	rho = np.zeros(face_seg.shape)
	rho = rho[:,:,0]

	roughness = np.zeros(face_seg.shape)
	roughness = roughness[:,:,0]

	for i in range(0,9):
		index = np.array(face_seg) == np.array(segments_color[i])

		rho[np.all(index, axis=-1)] = indv_specular_param[i, 0]
		roughness[np.all(index, axis=-1)] = indv_specular_param[i, 1]

		rho = rho.astype(np.float32)
		roughness = roughness.astype(np.float32)
		cv2.imwrite('./specularity_maps/rho/' + str(face_id) + '_rho.exr', cv2.GaussianBlur(rho,(63,63), 11))
		cv2.imwrite('./specularity_maps/roughness/' + str(face_id) + '_roughness.exr', cv2.GaussianBlur(roughness,(63,63), 11))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Optional app description')
	parser.add_argument('path_ethmodel', type=str,
	                    help='')
	args = parser.parse_args()

	# We marked segments on the facescape texture map
	# similar to MERL/ETH
	face_seg = cv2.imread('./instance_segments.png')
	segments_color = [[241, 239, 8], [15,7,244],[185, 26, 177],
					  [2, 251, 250], [251,0,0], [1,251,1],
					  [185,26,177], [129,126,0], [250,251,250]]
	
	# MERL/ETH Skin Reflectance Database
	subj_skintype, ts, ts_region, ts_subjec = load_eth_model(args.path_ethmodel)

	# (magic) offsets for facescape models to use MERL/ETH rs, m
	# rs -> rho, m -> roughness
	specular_hyperparam = [0.1, 0.2]
	os.makedirs('./specularity_maps/rho')
	os.makedirs('./specularity_maps/roughness')

	start = time.time()
	for face_id in range(1, len(subj_skintype)+1): # 156 subjects # range(10,152)?
		m, rs = initialize_parameters(subj_skintype, ts, ts_region, ts_subjec, face_id)

		indv_specular_param = compute_mean_parameters(m, rs, specular_hyperparam)
		save_map(indv_specular_param, face_seg, face_id)
	
	runtime = time.time() - start
	runtime /= 60
	print('Ellapsed time: %.1f minutes'%runtime)