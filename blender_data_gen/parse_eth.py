import scipy.io
# from collections import namedtuple
# import numpy as np
# from sklearn.cluster import MeanShift, estimate_bandwidth

'''
Load ETH skin reflectance model
'''

def load_eth_model(path):

	eth_skin_model = scipy.io.loadmat(path)

	# eth_captured_data = []
	# eth_faces = namedtuple('eth_faces', ['id', 'age', 'gender', 'stype'])

	# s_count = int(eth_skin_model['s_count'][0][0])
	# subj_age = eth_skin_model['subj_age'][0]
	# subj_gen = eth_skin_model['subj_gen'][0]
	subj_skintype = eth_skin_model['subj_skintype'][0]

	# for i in range(0, s_count):
	# 	eth_captured_data.append(eth_faces(i, int(subj_age[i]), subj_gen[i], int(subj_skintype[i])))

	ts = eth_skin_model['ts']
	ts_region = eth_skin_model['ts_region']
	ts_subjec = eth_skin_model['ts_subjec']

	# compare mean
	# albedo = eth_skin_model['albedo']
	# albedo_region = eth_skin_model['albedo_region']
	# albedo_subjec = eth_skin_model['albedo_subjec']

	return subj_skintype, ts, ts_region, ts_subjec


def initialize_parameters(subj_skintype, ts, ts_region, ts_subjec, face_id):

	print("Face ID: ", face_id, " skintype : ", subj_skintype[face_id-1])

	m = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
	rs = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
	for i in range(0, len(ts_subjec)):
		if ts_subjec[i,0] == face_id:
			m[ts_region[i,0]-1].append(ts[i,0])
			rs[ts_region[i,0]-1].append(ts[i,1])

	return m, rs



# '''
# Read FaceScape data info
# '''

# facescape_captured_data = []
# facescape_faces = namedtuple('facescape_faces', ['id', 'age', 'gender'])
# facescape_age = []

# with open('/data/jhtlab/FaceScape/info_list_v1.txt', 'r') as f:
# 	lines = f.readlines()
# 	for line in lines:
# 		index, gender, age = line.split(' ')
# 		if age != '-' and gender != '-':
# 			facescape_captured_data.append(facescape_faces(int(index), int(age), gender))

# def cluster_avg(x):
# 	if len(x) == 1:
# 		return 0
# 	X = list(zip(x,np.zeros(len(x))))
# 	X = np.asarray(X)
# 	bandwidth = estimate_bandwidth(X, quantile=0.3)
# 	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# 	ms.fit(X)
# 	labels = ms.labels_
# 	cluster_centers = ms.cluster_centers_

# 	labels_unique = np.unique(labels)
# 	n_clusters_ = len(labels_unique)

# 	max_len = -1000
# 	cluster_arr = []
# 	for k in range(n_clusters_):
# 		my_members = labels == k
# 		if len(X[my_members, 0]) > max_len:
# 			max_len = len(X[my_members, 0])
# 			cluster_arr = X[my_members, 0]

# 	return np.mean(cluster_arr)
