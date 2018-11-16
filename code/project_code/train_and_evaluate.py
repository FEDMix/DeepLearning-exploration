import operator
import cPickle as pickle
import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch import sigmoid
from torch import optim
from torch.autograd import Variable
import torchvision
torch.backends.cudnn.deterministic = True

from timeit import default_timer as timer
import random

from logger import Logger
from utils import *
from datasets import Promise2012_Dataset, Promise2012_Dataset_Pixelwise
from losses import CombinedLoss, BCELoss2d
from metrics import DiceCoeff
from nets import *

import sys
import matplotlib.pyplot as plt

def train_net(net,
              epochs=5,
              maximum_number_of_samples=10000,
              batch_size=1,
              lr=0.1,
              save_net='',
              gpu=True,
              image_dim=(512, 512),
              logging=False, logging_plots=False,
              eval_train=False,
              aggregate_epochs=1,
              start_evaluation_epoch=1, start_evaluation_samples=1000,
              train_patients=[],
              val_patients=[],
              dir_images='', dir_masks='', dir_checkpoints='', start_display_epoch=10,
              verbose=False, display_predictions=False, display_differences=False,
              MultiResolutionClassifier=False,
              dicom_format = False):

    set_random_seeds(gpu)

    torch.cuda.empty_cache()

    if logging:
        logger = Logger('../experiments/', {'dim': image_dim, 'learning_rate': lr},
                        track_variables=['train_bce', 'train_score', 'val_score', 'thousands_of_processed_samples'])

    if verbose:
        print('''
        Starting training:
            Epochs: %d
            Batch size: %d
            Learning rate: %f
            Training size: %d
            Validation size: %d
            Checkpoints: %s
            GPU: %s'''
              % (epochs, batch_size, lr, len(train_patients), len(val_patients), str(save_net), str(gpu)))

    N_train = len(train_patients)

    trainable_parameters = filter(lambda p: p.requires_grad, net.parameters())
    if verbose:
        print 'Number of trainable parameters sets %d' % len(trainable_parameters)
    
    if MultiResolutionClassifier:
        criterion = BCELoss2d()
    else:
        criterion = CombinedLoss(is_dice_log=False)
        
    if gpu:
        criterion = criterion.cuda()

    optimizer = optim.Adam(trainable_parameters, lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150, 200], gamma=0.1)

    total_number_of_samples = 0

    net.train()
    print type(net)
    if MultiResolutionClassifier:
        train_dataset = Promise2012_Dataset_Pixelwise(
            dir_images, dir_masks, train_patients, augment=True, gpu=gpu, image_dim=image_dim)
        train_dataset_without_augmentations = Promise2012_Dataset_Pixelwise(
            dir_images, dir_masks, train_patients, augment=False, gpu=gpu, image_dim=image_dim)
        val_dataset = Promise2012_Dataset_Pixelwise(
            dir_images, dir_masks, val_patients, augment=False, gpu=gpu, image_dim=image_dim)
    else:
        train_dataset = Promise2012_Dataset(
            dir_images, dir_masks, train_patients, augment=True, gpu=gpu, image_dim=image_dim)
        train_dataset_without_augmentations = Promise2012_Dataset(
            dir_images, dir_masks, train_patients, augment=False, gpu=gpu, image_dim=image_dim)
        val_dataset = Promise2012_Dataset(
            dir_images, dir_masks, val_patients, augment=False, gpu=gpu, image_dim=image_dim)
        
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4 if torch.cuda.device_count() > 1 else 1)
    train_loader_without_augmentations = torch.utils.data.DataLoader(dataset=train_dataset_without_augmentations,
                                                                     batch_size=batch_size,
                                                                     shuffle=False,
                                                                     pin_memory=True,
                                                                     num_workers=4 if torch.cuda.device_count() > 1 else 1)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=4 if torch.cuda.device_count() > 1 else 1)

    filename_masks_probs = {}
    eval_scores, best_eval_scores = {}, {}
    
    start_training = timer()

    try:
        for epoch in range(epochs):
            
            if verbose:
                print('Starting epoch %d/%d' % (epoch + 1, epochs))
            
            epoch_loss, epoch_bce_loss, epoch_dice_loss = 0.0, 0.0, 0.0

            net.train()         
            
            start_epoch_time = timer()

            start_epoch_total_number_of_samples = total_number_of_samples

            iters_count = 0
            for iter, b in enumerate(train_loader):
             
                if MultiResolutionClassifier:
                    images1, images2, images3, true_masks, filenames, index = b
                    #print len(index),index[0].shape
                else:    
                    images, true_masks, filenames = b
                
                if gpu:
                    if MultiResolutionClassifier:
                        images1 = Variable(images1.cuda())
                        images2 = Variable(images2.cuda())
                        images3 = Variable(images3.cuda())
                        true_masks = Variable(true_masks.cuda())
                        
                        masks_probs = net(images1, images2, images3)
                        
                    else:
                        images, true_masks = Variable(
                            images.cuda()), Variable(true_masks.cuda())
                        
                        masks_probs = net(images)                              

                masks_probs_flat = masks_probs.view(-1)
                true_masks_flat = true_masks.view(-1)
                
                if MultiResolutionClassifier:
                    loss = criterion(masks_probs_flat, true_masks_flat)
                    epoch_loss += loss.item()
                    #print masks_probs_flat, true_masks_flat, loss

                else:
                    loss, bce_loss, dice_loss = criterion(
                        masks_probs_flat, true_masks_flat)
                    epoch_loss += loss.item()
                    epoch_bce_loss += bce_loss.item()
                    epoch_dice_loss += dice_loss.item()
                    
                #print '%d --- loss: %.3f' % (iter, loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_number_of_samples += true_masks.shape[0]

                iters_count += 1

            if total_number_of_samples > maximum_number_of_samples:
                break

            if (total_number_of_samples // 1000) != (start_epoch_total_number_of_samples // 1000):
                scheduler.step()

            epoch_to_display = (epoch + 1) % (1 + (aggregate_epochs // 2)) == 0


            if (epoch + 1) >= start_evaluation_epoch or total_number_of_samples >= start_evaluation_samples:
                mean_eval_score, eval_scores = evaluate_net(net, val_loader,
                                                       gpu=gpu, batch_size=batch_size, 
                                                       display_predictions=(display_predictions and epoch >= start_display_epoch),
                                                       display_differences=(display_differences and epoch >= start_display_epoch),
                                                       aggregate_epochs=aggregate_epochs, 
                                                       filename_masks_probs=filename_masks_probs,
                                                           MultiResolutionClassifier=MultiResolutionClassifier,
                                                           dicom_format = dicom_format)

                if epoch_to_display:
                    if verbose:
                        print('Epoch %d, Validation Score: %.4f' %
                              (epoch + 1, mean_eval_score))
                    if logging:
                        if mean_eval_score > logger.get_best_value('val_score', 'max'):
                            best_eval_scores = eval_scores
                            
                            if len(save_net) > 0:
                                torch.save(net, os.path.join(dir_checkpoints, save_net))
                                print('Saved net %.3f' % mean_eval_score)
        
                        logger.update('val_score', mean_eval_score)
                        logger.update('thousands_of_processed_samples', total_number_of_samples//1000)
                        if logging_plots:
                          logger.plot_two_variables('val_score', 'thousands_of_processed_samples')
                        logger.save()
                        if verbose:
                            print('Best Validation Score %.4f' %
                                  logger.get_best_value('val_score', 'max'))
                
                if eval_train:
                    mean_eval_train_score, eval_train_scores = evaluate_net(net, train_loader_without_augmentations,
                                                       gpu=gpu, batch_size=batch_size, 
                                                       display_predictions=(display_predictions and epoch >= start_display_epoch),
                                                       display_differences=(display_differences and epoch >= start_display_epoch),
                                                       aggregate_epochs=aggregate_epochs, 
                                                       filename_masks_probs=filename_masks_probs,
                                                                           MultiResolutionClassifier=MultiResolutionClassifier,
                                                                           dicom_format = dicom_format)
                    
                    if epoch_to_display:
                        if verbose:
                            print('Epoch %d, Train Score: %.4f' %
                                  (epoch + 1, mean_eval_train_score))
                        if logging:
                            logger.update('train_score', mean_eval_train_score)
                            if logging_plots:
                              logger.plot_two_variables('train_score', 'thousands_of_processed_samples')
                            logger.save()
                            if verbose:
                                print('Best Train Score %.4f' %
                                      logger.get_best_value('train_score', 'max'))

    
            end_epoch_time = timer()

            if verbose:
                print 'Epoch %d finished ! Combined Loss: %.3f' % (epoch + 1, epoch_loss / iters_count)
                print 'Epoch %d finished ! BCE Loss: %.3f' % (epoch + 1, epoch_bce_loss / iters_count)
                print 'Epoch %d finished ! Score: %.3f' % (epoch + 1, -epoch_dice_loss / iters_count)
                print('Epoch time: %.1f seconds' %
                      (end_epoch_time - start_epoch_time))
                print('Number of processed samples: %d' %
                      total_number_of_samples)

            # if epoch_to_display and save_cp:
            #     filename = '%d_%.3f.pth' % (epoch+1, mean_eval_score)
            #     torch.save(net.state_dict(), os.path.join(dir_checkpoints, filename))
            #     print('Checkpoint at epoch %d with validation score %.3f saved !' % (epoch + 1, mean_eval_score))

    except KeyboardInterrupt:
        if len(save_net) > 0:
            torch.save(net, os.path.join(
                dir_checkpoints, 'INTERRUPTED_' + save_net))
            print('Saved interrupted')
            sys.exit(0)

    end_training = timer()
    training_time = end_training - start_training
    if verbose:
        print('Training time: %.1f seconds' % training_time)
        print('Number of processed samples: %d' % total_number_of_samples)

    if len(save_net) > 0:
        torch.save(net, os.path.join(dir_checkpoints, save_net.replace('.pth','')+'_final.pth'))
        print('Saved final net')
        
    if not logging:
        return eval_scores, net
    else:
        return best_eval_scores, net

def evaluate_net(net, eval_loader, gpu=False, batch_size=1, display_predictions=False, display_differences=False, aggregate_epochs=1, filename_masks_probs={}, MultiResolutionClassifier=False, dim = (256, 256),dicom_format = False):

    net.eval()

    filename_score = {}
    filename_masks_true = {}
    diffs = []
    
    start_eval_time = timer()
    
    for iter, b in enumerate(eval_loader):

        if MultiResolutionClassifier:
            images1, images2, images3, true_masks, filenames, index = b
            images_true = images1.float().detach()
        else:    
            images, true_masks, filenames = b
            images_true = images.float().detach()
            
        if gpu:
            if MultiResolutionClassifier:
                images1 = Variable(images1.cuda())
                images2 = Variable(images2.cuda())
                images3 = Variable(images3.cuda())
                true_masks = Variable(true_masks.cuda())

                masks_probs = net(images1, images2, images3)

            else:
                images, true_masks = Variable(
                    images.cuda()), Variable(true_masks.cuda())

                masks_probs = net(images)  
        
        
        masks_probs =  masks_probs.float().detach()
        true_masks = true_masks.float().detach()
        
        for i in range(masks_probs.shape[0]):
            
            if aggregate_epochs > 1:
                
                if filenames[i] not in filename_masks_probs:
                    filename_masks_probs[filenames[i]] = masks_probs[i].unsqueeze(0).clone()
                else:
                    filename_masks_probs[filenames[i]] = torch.cat(
                        (filename_masks_probs[filenames[i]], masks_probs[i].unsqueeze(0)), dim=0)  # add to end of the list

                # print filename_masks_probs[filenames[i]].shape
                if filename_masks_probs[filenames[i]].shape[0] > aggregate_epochs:
                    # pop from the beginning of the list
                    filename_masks_probs[filenames[i]] = filename_masks_probs[filenames[i]][1:]

                aggregated_prediction = torch.mean(
                    filename_masks_probs[filenames[i]], dim=0).cuda()
                
                cur_score = DiceCoeff(aggregated_prediction, true_masks[i])
                s1 = torch.sum((true_masks[i]>0.5).float()).cpu().item()
                s2 = torch.sum((aggregated_prediction>0.5).float()).cpu().item()
                filename_score[filenames[i]] = cur_score
                
                if cur_score < 0.8:
                    if s1 != 0:
                        diff = float(s2) / s1
                    else:
                        diff = 2

                    diffs.append(diff)
                
                if display_predictions:
                    if not dicom_format:
                        show_image(true_masks[i], title = 'ground_truth')
                        grid = torchvision.utils.make_grid(
                            filename_masks_probs[filenames[i]], nrow=5, padding=10, normalize=True)
                        show_grid(grid, title = 'predictions')
                    else:
                        patient = int(filenames[i].split('_')[0].replace('Case',''))
                        slice = int(filenames[i].split('_')[1].replace('.jpg',''))
                        #save_dicom(true_masks[i].detach().cpu().numpy(), 'true_mask_%d_%d' % (patient, slice), patient, slice)
                        filename = 'output/%d_%d.png' % (patient, slice)
                        save_prediction(images_true[i], true_masks[i], aggregated_prediction, filename, cur_score)
                            
                if display_differences:
                    if not dicom_format:
                        grid = torchvision.utils.make_grid(
                            true_masks[i]-filename_masks_probs[filenames[i]], nrow=5, padding=10, normalize=True)
                        show_grid(grid, title = 'ground truth minus predictions')

            else:
                if MultiResolutionClassifier:
                    if filenames[i] not in filename_score:
                        filename_masks_true[filenames[i]] = torch.tensor(np.zeros((1, dim[0], dim[1]), dtype = np.float32)).cuda()
                        filename_masks_probs[filenames[i]] = torch.tensor(np.zeros((1, dim[0], dim[1]), dtype = np.float32)).cuda()
                        filename_score[filenames[i]] = 0
                    filename_masks_probs[filenames[i]][0, index[0][i], index[1][i]] = masks_probs[i]
                    filename_masks_true[filenames[i]][0, index[0][i], index[1][i]] = true_masks[i]
                else:
                    cur_score = DiceCoeff(masks_probs[i], true_masks[i])
                    filename_score[filenames[i]] = cur_score
                
                if display_predictions:
                    if not MultiResolutionClassifier:
                        show_image(true_masks[i], title = 'ground_truth')
                if display_differences:
                    if not MultiResolutionClassifier:
                        show_image(true_masks[i]-filename_masks_probs[filenames[i]], title = 'ground truth minus predictions')

    if MultiResolutionClassifier:             
        for file in filename_masks_probs:     
            filename_score[file] = DiceCoeff(filename_masks_probs[file], filename_masks_true[file])
            print file, filename_score[file]
            if display_predictions:
                show_image(filename_masks_true[file], title = file + 'true')
                show_image(filename_masks_probs[file], title = file + 'pred')
                 
    torch.cuda.empty_cache()
    diffs = np.array(diffs).astype(np.float32)
    #plt.hist(diffs, bins = 20)
    #plt.show()
    print 'eval time: ', timer() - start_eval_time
    print 'over segmentation cases: %d | under segmentation cases: %d' % (np.where(diffs>1)[0].shape[0], np.where(diffs<1)[0].shape[0])

            
    return np.mean(filename_score.values()), filename_score
