'''
This script renders various reflectance components.
USAGE : 
./[blender_path]/blender --background ./render.blend --factory-startup \
--python blender_datagen.py -- --path_fs ../facescape_trainset_1_100 \
--path_hdr ./envmaps/ --output_dir ./train/ \
--modes ["Albedo","Diffuse","Specular","Normal","Mask","Final_Without_SSS"]
--indv [face_id] --hdr_name [envmap file name] --imageid [imageid]
'''

import bpy
from bpy import context, data, ops

import sys
import os

import argparse 
import math
import random
import time
import json

'''
	writes this to info.json for each generated output face.
'''

TEMPLATE_INFO = { 
	"FaceScape_id" : 0,
	"FaceScape_expression" : "",
	"face_position" : 0, 
	"face_rotation" : 0, 
	"camera_position" : 0, 
	"camera_rotation" : 0,
	"envmap_name" : "",
}

'''
	Expressions in the facescape dataset.
	We only used the neutral face in the paper.
'''

FACESCAPE_EXPS = [
				  "1_neutral", "2_smile", "3_mouth_stretch", "4_anger", "5_jaw_left", "6_jaw_right", 
				  "7_jaw_forward", "8_mouth_left", "9_mouth_right", "10_dimpler", "11_chin_raiser", 
				  "12_lip_puckerer", "13_lip_funneler", "14_sadness", "15_lip_roll", "16_grin", 
				  "17_cheek_blowing", "18_eye_closed", "19_brow_raiser", "20_brow_lower"
				]
'''

Define number of samples for each component

'''

NUM_SAMPLES = {
				"Albedo": 3, "Diffuse" : 700, "Specular" : 1,
				"Normal" : 3, "Depth" : 3, "Final_With_SSS" : 1,
				"Final_Without_SSS" : 1000, "Mask" : 3, "SSS_Diffuse" : 1,
				"Rho_map": 3, "Roughness_map": 3
			}

class Blender():
	def __init__(self):
		self.model = None
		self.ob = None
		self.width = 512
		self.height = 512

	def use_gpu(self):
		'''
			Set device to GPU
		'''
		bpy.context.scene.cycles.device = 'GPU'
		prefs = bpy.context.preferences
		cprefs = prefs.addons['cycles'].preferences
		cprefs.compute_device_type = 'CUDA'
		cprefs.get_devices()
		for device in cprefs.devices:
			device.use = True
		bpy.ops.wm.save_userpref()

	def import_model(self, path):
		'''
			Imports .obj model
		'''
		old_obj = set(context.scene.objects)
		bpy.ops.import_scene.obj(filepath=path, filter_glob="*.obj")
		self.model = (set(context.scene.objects) - old_obj).pop()
		self.ob = bpy.data.objects[0]
		print("Imported Geometry.............")

	def remove_object(self):
		bpy.ops.object.select_all(action='DESELECT')
		bpy.data.objects[0].select_set(True)
		bpy.ops.object.delete()
		print("Removed Geometry.............")

	def select_object(self):
		self.ob = bpy.data.objects[0]
		self.ob.select_set(True)

	def transform_model(self, indv):
		'''
			Scales the loaded object,
			applies smoothing to it,
			applies rotation to the object
		'''
		self.select_object()
		# Apply smooth shading
		mesh = self.ob.data
		for f in mesh.polygons:
			f.use_smooth = True
		# Scale down face
		self.ob.scale = (0.02, 0.02, 0.02)
		# Rotate face
		random.seed(indv)
		rot_x = random.randint(80, 100)
		# rot_y = random.randint(-20, 30)
		random.seed(indv)
		rot_z = random.randint(-10, 10)
		self.ob.rotation_euler = (math.radians(rot_x), 0, math.radians(rot_z))
		'''
			Saves these random transformations into info.json
		'''

		TEMPLATE_INFO["face_position"] = [self.ob.location[0], 
											self.ob.location[1],
											self.ob.location[2]]
		TEMPLATE_INFO["face_rotation"] = [self.ob.rotation_euler[0],
											self.ob.rotation_euler[1],
											self.ob.rotation_euler[2]]
		TEMPLATE_INFO["camera_position"] = [bpy.data.objects['Camera'].location[0],
												bpy.data.objects['Camera'].location[1],
												bpy.data.objects['Camera'].location[2]]
		TEMPLATE_INFO["camera_rotation"] = [bpy.data.objects['Camera'].rotation_euler[0],
												bpy.data.objects['Camera'].rotation_euler[1],
												bpy.data.objects['Camera'].rotation_euler[2]]
		print("Transformed Geometry.............")

	def load_envmap(self, hdr):
		'''
			Loads a random indoor env map
		'''
		for image in bpy.data.images:
			if image.name.split('.')[-1] == 'exr':
				bpy.data.images.remove(image)
		bpy.data.images.load(hdr, check_existing=False)
		bpy.context.scene.world.node_tree.nodes["Environment Texture"].image = bpy.data.images[hdr.split('/')[-1]]
		TEMPLATE_INFO["envmap_name"] = hdr
		print("loaded Environment Map.............")

	def set_material(self, material):
		'''
			Set the pre-defined materials: Diffuse albedo, specular, normal etc.
		'''
		self.select_object()
		context.view_layer.objects.active = self.ob
		if material in bpy.data.materials:
			mat = bpy.data.materials[material]
			if self.ob.data.materials:
				self.ob.data.materials[0] = mat
			else:
				self.ob.data.materials.append(mat)
		print("Assigned materials")

	def set_albedo(self, albedo_path, albedo_name, material):
		'''
			Load albedo texture map
		'''
		for image in bpy.data.images:
			if image.name.split('.jpg')[0] == albedo_name:
				bpy.data.images.remove(image)
		bpy.data.images.load(albedo_path, check_existing=False)
		mat = bpy.data.materials[material]
		bsdf = mat.node_tree.nodes['Image Texture']
		bsdf.image = bpy.data.images[albedo_name]
		print("Loaded Albedo GT.............")

	def set_specularity(self, material, indv):
		'''
			Load the precomputed ETH specular reflectance parameter from an image
		'''
		print("Assigning Specularity")
		random.seed(indv)
		idx = random.randint(1, 156) # merl/eth subjects
		roughness = str(idx) + '_roughness.exr'
		rho = str(idx) + '_rho.exr'
		for image in bpy.data.images:
			try:
				if image.name.split('.png')[0] == rho or image.name.split('.exr')[0] == rho:
					bpy.data.images.remove(image)
				if image.name.split('.png')[0] == roughness or image.name.split('.exr')[0] == roughness:
					bpy.data.images.remove(image)
			except:
				print('Cannot delete image.....')
		print("Removed Old Specular GT.............")
		bpy.data.images.load('./specularity_maps/roughness/' + roughness, check_existing=False)
		bpy.data.images.load('./specularity_maps/rho/' + rho, check_existing=False)
		mat = bpy.data.materials[material]
		mat.node_tree.nodes['Roughness'].image = bpy.data.images[roughness]
		mat.node_tree.nodes['Specularity'].image = bpy.data.images[rho]
		print("Loaded Specular GT.............")


	def set_specular_param(self, material, indv):
		'''
			Load the precomputed ETH specular reflectance parameter from an image
			This is used to render the maps as an albedo
		'''
		print("Assigning Specular Maps")
		random.seed(indv)
		idx = random.randint(1, 156) # merl/eth subjects
		if material == "Rho_map":
			param_map = str(idx) + '_rho.exr'
			map_path = 'rho'
		elif material == "Roughness_map":
			param_map = str(idx) + '_roughness.exr'
			map_path = 'roughness'
		for image in bpy.data.images:
			try:
				if image.name.split('.png')[0] == param_map or image.name.split('.exr')[0] == param_map:
					bpy.data.images.remove(image)
			except:
				print('Cannot delete image.....')
		print("Removed Old Specular GT.............")
		bpy.data.images.load('./specularity_maps/' + map_path + '/' + param_map, check_existing=False)
		mat = bpy.data.materials[material]
		bsdf = mat.node_tree.nodes['Image Texture']
		bsdf.image = bpy.data.images[param_map]
		print("Loaded Map as Albedo.............")

	def render_components(self, indv, modes, output_pth, 
						  expname, albedo_path, albedo_name):
		for mode in modes:
			self.set_material(mode)
			# Remove self-occlusions
			bpy.context.object.cycles_visibility.shadow = False
			# Toggle HDR background (TODO: turn off alpha)
			if mode in ["Final_With_SSS", "Final_Without_SSS"]:
				bpy.context.scene.render.film_transparent = True
			else:
				bpy.context.scene.render.film_transparent = True
			# Load Albedo
			if mode in ["Final_With_SSS", "Final_Without_SSS", "Albedo", "Diffuse", "SSS_Diffuse"]:
				self.set_albedo(albedo_path, albedo_name, mode)
			if mode in ["Final_With_SSS", "Final_Without_SSS", "Specular"]:
				self.set_specularity(mode, indv)
			if mode in ["Rho_map", "Roughness_map"]:
				self.set_specular_param(mode, indv)
			self.save_render(NUM_SAMPLES[mode], mode, output_pth, expname)

	def save_render(self, samples, mode, 
					output_pth, expname):
		self.use_gpu()
		context.scene.cycles.device = 'GPU'
		context.scene.cycles.samples = samples
		context.scene.render.resolution_x = self.width
		context.scene.render.resolution_y = self.height
		context.scene.render.resolution_percentage = 100
		bpy.context.scene.node_tree.nodes["File Output"].base_path = output_pth
		# bpy.context.scene.render.image_settings.color_mode = 'RGB'
		bpy.ops.render.render() # write_still=True
		os.system('mv ' + output_pth + '/openexr0003.exr ' + output_pth + '/' + expname + '_' + mode + '.exr')
		print("Rendered Image.............")
                # Save info to JSON file
		with open(output_pth + '/info.json', 'w') as fp:
			json.dump(TEMPLATE_INFO, fp)

		# Compute SSS contribution by subtracting diffuse from SSS (rendered without specular)

		# if mode == "Final_With_SSS":
		# 	sss = cv2.imread(output_pth + '/' + name + '_Final_With_SSS.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		# 	diffuse = cv2.imread(output_pth + '/' + name + '_Final_Without_SSS.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		# 	diff = abs(sss-diffuse) * 90000
		# 	cv2.imwrite(output_pth + '/' + name + '_diff.png', diff)


def compute_dmfp(A, sigma_a, sigma_s):
	d = 1 / ( (3.5 + 100 * pow((A - 0.33), 4)) *  pow(3 * sigma_a * (sigma_a + 0.3 * sigma_s), 0.5) )
	return d

def execute_blender(path_fs, path_hdr, output_dir,
					modes, num_exp, resolution,
					indv, HDR_name, imageid):
	'''
		Loads a face from the facescape directory
		and renders out the necessary components
	'''

	blender_instance = Blender()
	blender_instance.remove_object()

	blender_instance.width = int(resolution.split('x')[0])
	blender_instance.height = int(resolution.split('x')[1])

	modes = modes[1:-1]
	modes = modes.split(',')

	indv = str(indv)
	output_pth = os.path.join(output_dir, imageid)
	os.system('mkdir ' + output_pth)
	for exp in range(0, num_exp):
		try:
			obj_path = os.path.join(path_fs, indv, 'models_reg', str(FACESCAPE_EXPS[exp]) + '.obj')
			albedo_path = os.path.join(path_fs, indv, 'models_reg', str(FACESCAPE_EXPS[exp]) + '.jpg')
			TEMPLATE_INFO["FaceScape_id"] = indv
			TEMPLATE_INFO["FaceScape_expression"] = FACESCAPE_EXPS[exp]
			# Import OBJ
			blender_instance.import_model(obj_path)
			# Transform the imported face
			blender_instance.transform_model(indv)
			# Load Env Map
			HDR = path_hdr + HDR_name
			blender_instance.load_envmap(HDR)
			# Render components
			blender_instance.render_components(indv, modes, output_pth, str(exp + 1), 
											   albedo_path, str(FACESCAPE_EXPS[exp]) + '.jpg')
			# Remove object
			blender_instance.remove_object()
		except:
			print("An error occured when execute blender rendering!!!")
		break
		

def main():

	argv = sys.argv

	if "--" not in argv:
		argv = []  # as if no args are passed
	else:
		argv = argv[argv.index("--") + 1:]  # get all args after "--"

	# When --help or no args are given, print this help
	usage_text = (
		"Run blender in background mode with this script:"
		"  blender --background --python " + __file__ + " -- [options]"
	)

	parser = argparse.ArgumentParser(description=usage_text)

	parser.add_argument(
		"-fs", "--path_fs", dest="path_fs", type=str, required=True,
		help="Path to facescape capture set",
	)

	parser.add_argument(
		"-hdr", "--path_hdr", dest="path_hdr", type=str, required=True,
		help="Path to Env Maps",
	)

	parser.add_argument(
		"-output", "--output_dir", dest="output_dir", type=str, required=True,
		help="Path to output dir",
	)

	parser.add_argument(
		"-modes", "--modes", dest="modes", type=str, required=True,
		help="List containing the components to be rendered, " +
		" eg: [\"Albedo\",\"Diffuse\",\"Specular\",\"Normal\",\"Mask\",\"Depth\",\"Final_With_SSS\",\"Final_Without_SSS\"]",
	)

	parser.add_argument(
		"-num_exp", "--num_exp", dest="num_exp", type=int, required=False,
		default=1, help="No. of expressions to be rendered per face, total = 20",
	)


	parser.add_argument(
		"-r", "--resolution", dest="resolution", type=str, required=False,
		default="512x512", help="Output resolution",
	)

	parser.add_argument(
		"-indv", "--indv", dest="indv", type=str, required=True,
		default="1", help="facescape individual id",
	)

	parser.add_argument(
		"-hdr_name", "--hdr_name", dest="hdr_name", type=str, required=True,
		 help="HDR name",
	)

	parser.add_argument(
		"-imageid", "--imageid", dest="imageid", type=str, required=True,
		 default='0', help="imageid for saving the images",
	)

	'''
	This doesn't work. It is a bug in bpy. So you'll have to manually set the device to use.
	By default it should be the GPU.
	'''

	parser.add_argument(
		"-d", "--device", dest="device", type=str, required=False,
		default="GPU", help="CPU / GPU",
	)

	args = parser.parse_args(argv)

	if not argv:
		parser.print_help()
		return

	if (not args.path_fs or 
		not args.path_hdr or 
		not args.output_dir or
		not args.modes or
		not args.indv or
		not args.hdr_name):
		print("Error: argument not given, aborting.")
		parser.print_help()
		return

	os.system('mkdir ' + args.output_dir)

	start = time.time()
	execute_blender(args.path_fs, args.path_hdr, args.output_dir, args.modes, \
		args.num_exp, args.resolution, args.indv, args.hdr_name, args.imageid)

	print(time.time() - start)

if __name__ == "__main__":
	main()
