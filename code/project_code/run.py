import numpy as np
import sys

import skimage.io as io
import os
import numpy as np
from skimage import exposure
from scipy.stats import norm
import matplotlib.pyplot as plt

from nets import *
from train_and_evaluate import *

from timeit import default_timer as timer
import json

dir_images = '/export/scratch1/home/arkadiy/fedmix/prostate_challenge/data/images/'
dir_masks = '/export/scratch1/home/arkadiy/fedmix/prostate_challenge/data/segmentations_with_variation_mixed/'
dir_checkpoints = '/export/scratch1/home/arkadiy/fedmix/prostate_challenge/checkpoints/'
dir_evaluations = '/export/scratch1/home/arkadiy/fedmix/prostate_challenge/evaluations'

MIN_VAL_COUNT = 50
N_TRAIN_PATIENTS = 10

def run_training(array):
    
    try:
        print("received array", array)

        N_PATIENTS = len(array)

        patients_parts = [[],[]]
        patients_parts[0] = [i for i in range(N_PATIENTS) if array[i] == 1]
        patients_parts[1] = [i for i in range(N_PATIENTS) if array[i] == 0]

        print( 'parts:', patients_parts)

        np.random.seed(1)
        total_result_all_parts = 0.0

        for part_index in range(len(patients_parts)):

            current_patients = patients_parts[part_index]
            current_evaluations_files = os.listdir(dir_evaluations)
            current_evaluations_number = 0
            
            current_evaluations_results = {}
            current_evaluations_count = {}
            for patient in current_patients:
                current_evaluations_results[patient] = 0.0
                current_evaluations_count[patient] = 0

            for file in current_evaluations_files:
                evaluation_results = json.load(open(os.path.join(dir_evaluations, file),'r'))
                print('found set', evaluation_results['train_patients'], 'current set: ', current_patients)
                if set(evaluation_results['train_patients']) < set(current_patients):
                    for patient in evaluation_results['val_patients']:
                        if int(patient) in current_patients:
                            print(patient, evaluation_results['val_patients'][patient])
                            current_evaluations_results[int(patient)] += evaluation_results['val_patients'][patient]
                            current_evaluations_count[int(patient)] += 1
            
            print(current_evaluations_count, '\n\n', current_evaluations_results)
        
            while np.sum(current_evaluations_count.values()) < MIN_VAL_COUNT:
                
                print(np.sum(current_evaluations_count.values())

                np.random.seed(int(timer()))
                train_patients = np.random.permutation(current_patients)[:N_TRAIN_PATIENTS]
                val_patients = [patient for patient in range(N_PATIENTS) if patient not in train_patients]
                print(train_patients, val_patients)

                net = UNet_light(n_channels=1, n_classes=1)
                net = torch.nn.DataParallel(net).cuda()

                val_dices, net = train_net(net=net,
                                           epochs=300,
                                           maximum_number_of_samples=80000,
                                           batch_size=64,
                                           lr=0.001,
                                           gpu=True,
                                           image_dim=(256, 256),
                                           save_net='',
                                           aggregate_epochs=20,
                                           start_evaluation_epoch=1000, start_evaluation_samples=50000,
                                           verbose=True,
                                           display_predictions=False,
                                           display_differences=False,
                                           logging=True, logging_plots=False,
                                           eval_train=False,
                                           train_patients=train_patients,
                                           val_patients=val_patients,
                                           dir_images=dir_images,
                                           dir_masks=dir_masks,
                                           dir_checkpoints=dir_checkpoints)
                
                val_dices_by_patient = {'train_patients': list(train_patients), 'val_patients':{}}
                for patient in val_patients:
                    cur_patient_scores = [slice for slice in val_dices if int(slice[4:6]) == patient]
                    patient_score = 0.0
                    for slice in cur_patient_scores:
                        patient_score += val_dices[slice]
                    patient_score /= len(cur_patient_scores)
                    val_dices_by_patient['val_patients'][patient] = patient_score

                for patient in val_dices_by_patient['val_patients']:
                    if patient in current_evaluations_results:
                        current_evaluations_results[patient] += val_dices_by_patient['val_patients'][patient]
                        current_evaluations_count[patient] += 1

                print(current_evaluations_results, current_evaluations_count)
                json.dump(val_dices_by_patient, open(os.path.join(dir_evaluations, '%d.json' % len(os.listdir(dir_evaluations))),'w'))

            total_result = 0.0
            non_zero_patients = 0
            for patient in current_evaluations_count:
                if current_evaluations_count[patient]:
                    total_result += current_evaluations_results[patient] / current_evaluations_count[patient]
                    non_zero_patients += 1

            total_result /= non_zero_patients
            print(total_result)
            total_result_all_parts += total_result

        total_result_all_parts /= len(patients_parts)
        return total_result_all_parts

    except Exception as e:
        print(e)
        return 0.0

if __name__ == '__main__':
    run_training([1 for i in range(20)] + [0 for i in range(30)])
