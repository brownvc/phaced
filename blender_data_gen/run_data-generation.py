import os
import glob
import argparse
import random
import time


def main(blender_path, facescape_path, envmap_path, output_paths, resolution):

    facescape_ids_train = [i for i in range(1, 96)]
    facescape_ids_test = [i for i in range(96, 101)]
    HDRs = [os.path.basename(f) for f in glob.glob(
        os.path.join(envmap_path, "*.exr"))]
    HDRs.sort()
    # seeds for assigning different envmaps to face models in every full iteration
    seeds = [2002, 2003]
    count_train = 0
    count_test = 0
    for seed in seeds:
        random.seed(seed)
        random.shuffle(HDRs)

        for face_id in facescape_ids_train:
            os.system(blender_path + '/blender --background ./render.blend --factory-startup --python blender_datagen.py -- \
                    --path_fs ' + facescape_path + ' --path_hdr ' + envmap_path + ' --output_dir ' + output_paths[0] + ' --modes \
                        ["Albedo","Diffuse","Normal","Mask","Final_Without_SSS","Rho_map","Roughness_map"] --resolution ' + resolution +' -num_exp 1 --indv '
                        + str(face_id) + ' --hdr_name ' + str(HDRs[face_id % len(HDRs)]) + ' --imageid ' + str(count_train))
            count_train += 1
        for face_id in facescape_ids_test:
            os.system(blender_path + '/blender --background ./render.blend --factory-startup --python blender_datagen.py -- \
                    --path_fs ' + facescape_path + ' --path_hdr ' + envmap_path + ' --output_dir ' + output_paths[1] + ' --modes \
                        ["Albedo","Diffuse","Normal","Mask","Final_Without_SSS","Rho_map","Roughness_map"] --resolution ' + resolution +' -num_exp 1 --indv '
                        + str(face_id) + ' --hdr_name ' + str(HDRs[face_id % len(HDRs)]) + ' --imageid ' + str(count_test))
            count_test += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('blender_path', type=str,
                        help='blender_path')
    parser.add_argument('facescape_path', type=str,
                        help='facescape_path')
    parser.add_argument('envmap_path', type=str,
                        help='envmap_path')
    parser.add_argument('resolution', type=str, default='512x512', help='resolution')
    parser.add_argument('-out', '--out', nargs='+', default=['./train/', './test/'])

    args = parser.parse_args()
    # print(args.out)
    start = time.strftime('%X %x %Z')
    main(args.blender_path, args.facescape_path, args.envmap_path, args.out, args.resolution)
    end = time.strftime('%X %x %Z')
    print('------------------Data generation runtime---------------')
    print('Started: \t%s' % start)
    print('Ended: \t%s' % end)

# python run_data-generation.py ./blender-2.80-linux-glibc217-x86_64 ./facescape_trainset_1_100/ ../data/envmaps/ 128x128 -out ./train/ ./test/
