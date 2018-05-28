'''
This Program is developed to run pretrained model on a test dataset
'''
import torch
from torch.autograd import Variable
from preprocessing import *
from train import *
from bleu import *
import os
from machine_translation_vision.utils import im_retrieval_eval

use_cuda = torch.cuda.is_available()
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 80
batch_size = 64
eval_batch_size = 12
beam_size = 12
shared_embedding_size = 1024

#Generate the samples from validation dataset
#Load the Dataset
#data_path = '/home/zmykevin/machine_translation_vision/dataset/MMT_WMT_2017_MSCOCO_FR'
data_path = '/home/zmykevin/machine_translation_vision/dataset/Multi30K_FR_BPE_Kevin_Long'
#data_path = '/home/zmykevin/machine_translation_vision/dataset/Multi30K_DE_BPE_Kevin'
source_language = 'en'
target_language = 'fr'
vocab_path = '/home/zmykevin/machine_translation_vision/dataset/Multi30K_FR_BPE_Kevin'
BPE_dataset_suffix = '.subset.norm.tok.lc.10000bpe'
dataset_suffix = '.subset.norm.tok.lc'
dataset_im_suffix = '.subset.norm.tok.lc.10000bpe_ims'
model_folder = "/home/zmykevin/machine_translation_vision/code/mtv_trained_model/WMT17/nmt_FR_10_4"
model_path = os.path.join(model_folder,"nmt_wmt_baseline_trained_model_best_BLEU.pt")
loss_model_path = os.path.join(model_folder,"nmt_wmt_baseline_trained_model_best_loss.pt")
#meteor_model_path = os.path.join(model_folder,"nmt_trained_imagine_model_best_METEOR.pt")
#output_path = "/home/zmykevin/machine_translation_vision/code/mtv_trained_model/WMT17/nmt_imagine_FR_11"
output_path = "/home/zmykevin/machine_translation_vision/code/mtv_trained_model/Multi30K_Long_Subset/nmt_FR_10_4"
#output_path = "/home/zmykevin/machine_translation_vision/code/mtv_trained_model/MSCOCO17/nmt_FR_10_4"

#Create the directory for the trained_model_output_path
if not os.path.isdir(output_path):
    os.mkdir(output_path)
#Load the test dataset
test_source = load_data(os.path.join(data_path,'test'+BPE_dataset_suffix+'.'+source_language))
test_target = load_data(os.path.join(data_path,'test'+BPE_dataset_suffix+'.'+target_language))
print('The size of Test Source and Test Target is: {},{}'.format(len(test_source),len(test_target)))

#Load the original test dataset
test_ori_source = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+source_language))
test_ori_target = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+target_language))

#Create the paired test_data
test_data = [[x.strip(),y.strip()] for x,y in zip(test_source,test_target)]

#Creating List of pairs in the format of [[en_1,de_1], [en_2, de_2], ....[en_3, de_3]] for original data
test_ori_data = [[x.strip(),y.strip()] for x,y in zip(test_ori_source,test_ori_target)]

#Filter the data
test_data = data_filter(test_data,MAX_LENGTH)

#Filter the original data
test_ori_data = data_filter(test_ori_data,MAX_LENGTH)

print("The size of Test Data after filtering: {}".format(len(test_data)))


#Load the Vocabulary File and Create Word2Id and Id2Word dictionaries for translation
vocab_source = load_data(os.path.join(vocab_path,'vocab.'+source_language))
vocab_target = load_data(os.path.join(vocab_path,'vocab.'+target_language))


#Construct the source_word2id, source_id2word, target_word2id, target_id2word dictionaries
s_word2id, s_id2word = construct_vocab_dic(vocab_source)
t_word2id, t_id2word = construct_vocab_dic(vocab_target)

print("The vocabulary size for soruce language: {}".format(len(s_word2id)))
print("The vocabulary size for target language: {}".format(len(t_word2id)))

#Generate Train, Val and Test Indexes pairs
test_data_index = create_data_index(test_data,s_word2id,t_word2id)

test_y_ref = [[d[1].split()] for d in test_ori_data]

#Load the vision features
test_im_feats = np.load(os.path.join(data_path,'test'+dataset_im_suffix+'.npy'))

#Load the model
best_model = torch.load(model_path)

if use_cuda:
    best_model.cuda()

#Convert best_model to eval phase
best_model.eval()

test_translations = []
for test_x,test_y,test_x_lengths,test_y_lengths,test_sorted_index in data_generator(test_data_index,eval_batch_size):
    test_translation = best_model.beamsearch_decode(test_x,test_x_lengths,beam_size,MAX_LENGTH)

    #Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation,test_sorted_index,t_id2word) 
    test_translations += test_translation_reorder

#Compute the test bleu score
test_bleu = compute_bleu(test_y_ref,test_translations)
print("test_bleu from the best BLEU model: {}".format(test_bleu[0]))

#Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(output_path,'test_2017_prediction_best_BLEU.'+target_language)

with open(test_prediction_path,'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')

#Evalute the final results with nmtpy-coco-metrics
ground_truth_path = os.path.join(data_path,'test'+dataset_suffix+'.'+target_language)

print("Full evaluation results with best BLEU Model:")
#Execute nmtpy-coco-metrics to get the outputs
os.system('nmtpy-coco-metrics {} -l {} -r {}'.format(test_prediction_path,target_language,ground_truth_path))
###############################best_loss_model################################

best_loss_model = torch.load(loss_model_path)
if use_cuda:
    best_loss_model.cuda()

#Convert best_model to eval phase
best_loss_model.eval()

test_translations = []
for test_x,test_y,test_x_lengths,test_y_lengths,test_sorted_index in data_generator(test_data_index,eval_batch_size):
    test_translation = best_loss_model.beamsearch_decode(test_x,test_x_lengths,beam_size,MAX_LENGTH)

    #Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation,test_sorted_index,t_id2word) 
    test_translations += test_translation_reorder

#Compute the test bleu score
test_bleu = compute_bleu(test_y_ref,test_translations)
print("test_bleu from the best loss model: {}".format(test_bleu[0]))

#Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(output_path,'test_2017_prediction_best_loss.'+target_language)

with open(test_prediction_path,'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')

#Evalute the final results with nmtpy-coco-metrics
ground_truth_path = os.path.join(data_path,'test'+dataset_suffix+'.'+target_language)

print("Full evaluation results with best Loss model:")
#Execute nmtpy-coco-metrics to get the outputs
os.system('nmtpy-coco-metrics {} -l {} -r {}'.format(test_prediction_path,target_language,ground_truth_path))
