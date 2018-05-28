#This Process is Designed for the nmt_attentionimagine_sea2seq_BEAM_V11, where access to images during test is needed.
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from preprocessing import *
from machine_translation_vision.models import NMT_AttentionImagine_Seq2Seq_Beam_V11,\
                                              NMT_AttentionImagine_Seq2Seq_Beam_V7, \
                                              NMT_AttentionImagine_Seq2Seq_Beam_V12
from machine_translation_vision.losses import PairwiseRankingLoss
from machine_translation_vision.losses import ImageRetrievalRankingLoss
from machine_translation_vision.utils import im_retrieval_eval
from machine_translation_vision.meteor.meteor import Meteor

from train import *
from bleu import *

import time
import random
from random import randint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#The token index for the start of the sentence
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 80 #We will abandon any sentence that is longer than this length


use_cuda = torch.cuda.is_available()
print("Whether GPU is available: {}".format(use_cuda))

## Helper Functions to Print Time Elapsed and Estimated Time Remaining, give the current time and progress
def as_minutes(s):
    m = math.floor(s/60)
    s-= m*60
    return '%dm %ds'%(m,s)
def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)
    rs = es-s
    return '%s (- %s)'%(as_minutes(s),as_minutes(rs))

def save_plot(points,x_axis,save_path,y_label):
    plt.plot(x_axis,points)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.clf()

def save_plot_compare(point_1, point_2,label_1,label_2,x_axis, save_path, y_label):
    plt.plot(x_axis,point_1,label=label_1)
    plt.plot(x_axis,point_2,label=label_2)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.clf()

#############################################################Load the Dataset#######################################################
data_path = '/home/zmykevin/machine_translation_vision/dataset/Multi30K_FR_BPE_Kevin'
trained_model_output_path = '/home/zmykevin/machine_translation_vision/code/mtv_trained_model/WMT17/nmt_imagine_FR_16_4'
#trained_model_output_path = '/home/zmykevin/Kevin/Research/machine_translation_vision/trained_model/WMT17'
source_language = 'en'
target_language = 'fr'
BPE_dataset_suffix = '.norm.tok.lc.10000bpe'
dataset_suffix = '.norm.tok.lc'
dataset_im_suffix = '.norm.tok.lc.10000bpe_ims'
#Initilalize a Meteor Scorer
Meteor_Scorer = Meteor(target_language)

#Create the directory for the trained_model_output_path
if not os.path.isdir(trained_model_output_path):
    os.mkdir(trained_model_output_path)

#Load the training dataset
train_source = load_data(os.path.join(data_path,'train'+BPE_dataset_suffix+'.'+source_language))
train_target = load_data(os.path.join(data_path,'train'+BPE_dataset_suffix+'.'+target_language))

print('The size of Training Source and Training Target is: {},{}'.format(len(train_source),len(train_target)))

#Load the validation dataset
val_source = load_data(os.path.join(data_path,'val'+BPE_dataset_suffix+'.'+source_language))
val_target = load_data(os.path.join(data_path,'val'+BPE_dataset_suffix+'.'+target_language))
print('The size of Validation Source and Validation Target is: {},{}'.format(len(val_source),len(val_target)))

#Load the test dataset
test_source = load_data(os.path.join(data_path,'test'+BPE_dataset_suffix+'.'+source_language))
test_target = load_data(os.path.join(data_path,'test'+BPE_dataset_suffix+'.'+target_language))
print('The size of Test Source and Test Target is: {},{}'.format(len(test_source),len(test_target)))

#Load the original validation dataset
val_ori_source = load_data(os.path.join(data_path,'val'+dataset_suffix+'.'+source_language))
val_ori_target = load_data(os.path.join(data_path,'val'+dataset_suffix+'.'+target_language))

#Load the original test dataset
test_ori_source = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+source_language))
test_ori_target = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+target_language))

#Creating List of pairs in the format of [[en_1,de_1], [en_2, de_2], ....[en_3, de_3]]
train_data = [[x.strip(),y.strip()] for x,y in zip(train_source,train_target)]
val_data = [[x.strip(),y.strip()] for x,y in zip(val_source,val_target)]
test_data = [[x.strip(),y.strip()] for x,y in zip(test_source,test_target)]

#Creating List of pairs in the format of [[en_1,de_1], [en_2, de_2], ....[en_3, de_3]] for original data
val_ori_data = [[x.strip(),y.strip()] for x,y in zip(val_ori_source,val_ori_target)]
test_ori_data = [[x.strip(),y.strip()] for x,y in zip(test_ori_source,test_ori_target)]

#Filter the data
train_data = data_filter(train_data,MAX_LENGTH)
val_data = data_filter(val_data,MAX_LENGTH)
test_data = data_filter(test_data,MAX_LENGTH)

#Filter the original data
val_ori_data = data_filter(val_ori_data,MAX_LENGTH)
test_ori_data = data_filter(test_ori_data,MAX_LENGTH)

print("The size of Training Data after filtering: {}".format(len(train_data)))
print("The size of Val Data after filtering: {}".format(len(val_data)))
print("The size of Test Data after filtering: {}".format(len(test_data)))

#Load the Vocabulary File and Create Word2Id and Id2Word dictionaries for translation
vocab_source = load_data(os.path.join(data_path,'vocab.'+source_language))
vocab_target = load_data(os.path.join(data_path,'vocab.'+target_language))

#Construct the source_word2id, source_id2word, target_word2id, target_id2word dictionaries
s_word2id, s_id2word = construct_vocab_dic(vocab_source)
t_word2id, t_id2word = construct_vocab_dic(vocab_target)

print("The vocabulary size for soruce language: {}".format(len(s_word2id)))
print("The vocabulary size for target language: {}".format(len(t_word2id)))

#Generate Train, Val and Test Indexes pairs
train_data_index = create_data_index(train_data,s_word2id,t_word2id)
val_data_index = create_data_index(val_data,s_word2id,t_word2id)
test_data_index = create_data_index(test_data,s_word2id,t_word2id)

val_y_ref = [[d[1].split()] for d in val_ori_data]
test_y_ref = [[d[1].split()] for d in test_ori_data]

#Define val_y_ref_meteor and test_y_ref_meteor
val_y_ref_meteor = dict((key,[value[1]]) for key,value in enumerate(val_ori_data))
test_y_ref_meteor = dict((key,[value[1]]) for key,value in enumerate(test_ori_data))

#Load the Vision Features
train_im_feats = np.load(os.path.join(data_path,'train'+dataset_im_suffix+'.npy'))
val_im_feats = np.load(os.path.join(data_path,'val'+dataset_im_suffix+'.npy'))
test_im_feats = np.load(os.path.join(data_path,'test'+dataset_im_suffix+'.npy'))

#Verify the size of the train_im_features
print("Training Image Feature Size is: {}".format(train_im_feats.shape))
print("Validation Image Feature Size is: {}".format(val_im_feats.shape))
print("Testing Image Feature Size is: {}".format(test_im_feats.shape))


##############################Define Model and Training Structure##################################
#Network Structure
imagine_attn = "dot"
activation_vse = True
embedding_size = 256
hidden_size = 512
shared_embedding_size = 512
n_layers = 1
tied_emb = True

#Dropout
dropout_im_emb = 0.2
dropout_txt_emb = 0.0
dropout_rnn_enc = 0.0
dropout_rnn_dec = 0.0
dropout_emb = 0.2
dropout_ctx = 0.4
dropout_out = 0.4

#Training Setting
batch_size = 32
eval_batch_size = 16
batch_num = math.floor(len(train_data_index)/batch_size)
learning_rate = 0.001
weight_decay = 0.00001
loss_w= 0.99
beam_size = 12
n_epochs = 100
print_every = 100
eval_every = 1000
save_every = 10000
vse_separate = False
vse_loss_type = 'pairwise' #For model V8, we use a different loss called im_retrieval
#Define the teacher force_ratio
teacher_force_ratio = 0.8
clip = 1.0
#Define the margin size
margin_size = 0.1
patience = 10

#Initialize models
input_size = len(s_word2id)+1
output_size = len(t_word2id)+1

#Definet eh init_split
init_split = 0.5

#Define the model
imagine_model = NMT_AttentionImagine_Seq2Seq_Beam_V12(input_size, 
                                                  output_size,
                                                  train_im_feats.shape[1],
                                                  embedding_size, \
                                                  embedding_size, \
                                                  hidden_size, \
                                                  shared_embedding_size, \
                                                  loss_w, \
                                                  activation_vse=activation_vse, \
                                                  attn_model=imagine_attn, \
                                                  dropout_ctx=dropout_ctx, \
                                                  dropout_emb=dropout_emb, \
                                                  dropout_out=dropout_out, \
                                                  dropout_rnn_enc=dropout_rnn_enc, \
                                                  dropout_rnn_dec=dropout_rnn_dec, \
                                                  dropout_im_emb=dropout_im_emb, \
                                                  dropout_txt_emb=dropout_txt_emb, \
                                                  tied_emb=tied_emb,\
                                                  init_split=init_split)

if use_cuda:
    imagine_model.cuda()
#Use Multiple GPUs if they are available
"""
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(),"GPUs!")
    baseline_model = nn.DataParallel(baseline_model)
"""
#print(imagine_model)

#Define the loss criterion
vocab_mask = torch.ones(output_size)
vocab_mask[0] = 0
if use_cuda:
    vocab_mask = vocab_mask.cuda()

criterion_mt = nn.NLLLoss(weight=vocab_mask,reduce=False)
#criterion_vse = nn.HingeEmbeddingLoss(margin=margin_size,size_average=False)
if vse_loss_type == "pairwise":
  criterion_vse = PairwiseRankingLoss(margin=margin_size)
if vse_loss_type == "imageretrieval":
  criterion_vse = ImageRetrievalRankingLoss(margin=margin_size)

if use_cuda:
    criterion_vse = criterion_vse.cuda()
    criterion_mt = criterion_mt.cuda()


if not vse_separate:
    #Define the optimizer
    #optimizer = optim.Adam(imagine_model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    weight_group = { 'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' not in n],
                     'weight_decay': weight_decay,
    }

    bias_group = { 'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' in n],
        
    }

    param_groups = [weight_group,bias_group]
else:
    mt_weight_group = {'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' not in n and 'vse_imagine' not in n],
                     'weight_decay': weight_decay,
                     }
    mt_bias_group = { 'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' in n and 'vse_imagine' not in n],
        
    }
    vse_weight_group = {'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' not in n and 'vse_imagine' in n],
                     'weight_decay': weight_decay,
                     'lr': learning_rate/2,
                     }
    vse_bias_group = {'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, imagine_model.named_parameters())) if 'bias' in n and 'vse_imagine' in n],
                     'lr': learning_rate/2,
                     }
    param_groups = [mt_weight_group, mt_bias_group, vse_weight_group, vse_bias_group]

#Define Optimizer
optimizer = optim.Adam(param_groups,lr=learning_rate) #Optimize the parameters

#Define a learning rate optimizer
lr_decay_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=10)

################################Print the configuration settings#################################
print("Configurations:")
print('\n')
print("######## Network Structure #########")
print("embedding_size: {}".format(embedding_size))
print("hidden_size: {}".format(hidden_size))
print("shared_embedding_size: {}".format(shared_embedding_size))
print("n_layers: {}".format(n_layers))
print("tied_emb: {}".format(tied_emb))
print('\n')
print("####### Dropout #######")
print("dropout_im_emb: {}".format(dropout_im_emb))
print("dropout_txt_emb: {}".format(dropout_txt_emb))
print("dropout_rnn_enc: {}".format(dropout_rnn_enc))
print("dropout_rnn_dec: {}".format(dropout_rnn_dec))
print("dropout_emb: {}".format(dropout_emb))
print("dropout_ctx: {}".format(dropout_ctx))
print("dropout_out: {}".format(dropout_out))
print('\n')
print("####### Training Setting #######")
print("batch_size: {}".format(batch_size))
print("eval_batch_size: {}".format(eval_batch_size))
print("learning_rate: {}".format(learning_rate))
print("weight_decay: {}".format(weight_decay))
print("loss_w: {}".format(loss_w))
print("beam_size: {}".format(beam_size))
print("n_epochs: {}".format(n_epochs))
print("print_every: {}".format(print_every))
print("eval_every: {}".format(eval_every))
print("save_every: {}".format(save_every))
print("vse_separate: {}".format(vse_separate))
print("teacher_force_ratio: {}".format(teacher_force_ratio))
print("clip: {}".format(clip))
print("input_size: {}".format(input_size))
print("output_size: {}".format(output_size))
print("vse_margin: {}".format(margin_size))
print("vse_loss_type: {}".format(vse_loss_type))
print("init_split: {}".format(init_split))
print('\n')

##########################################Begin Training###########################################
#Print Loss
print_mt_loss = 0 #Reset every print_every
print_vse_loss = 0
print_loss = 0

#Start Training
print("Begin Training")
start = time.time()
iter_count = 0
best_bleu = 0
best_meteor = 0
best_loss = 10000000
early_stop = patience
for epoch in range(1,n_epochs + 1):
    for batch_x,batch_y,batch_im,batch_x_lengths,batch_y_lengths in data_generator_tl_mtv(train_data_index,train_im_feats,batch_size):
        #Run the train function
        train_loss,train_loss_mt,train_loss_vse = train_imagine_beam(batch_x,batch_y,batch_im,batch_x_lengths,imagine_model,optimizer,criterion_mt,criterion_vse,loss_w,teacher_force_ratio,clip=clip)
        
        print_loss += train_loss
        
        #Update translation loss and vse loss
        print_mt_loss += train_loss_mt
        print_vse_loss += train_loss_vse
        
        if iter_count == 0: 
            iter_count += 1
            continue
        
        if iter_count%print_every == 0:
            print_loss_avg = print_loss / print_every
            print_mt_loss_avg = print_mt_loss / print_every
            print_vse_loss_avg = print_vse_loss / print_every
            #Reset the print_loss, print_mt_loss and print_vse_loss
            print_loss = 0
            print_mt_loss = 0
            print_vse_loss = 0
            
            print_summary = '%s (%d %d%%) train_loss: %.4f, train_mt_loss: %.4f, train_vse_loss: %.4f' % (time_since(start, iter_count / n_epochs / batch_num), iter_count, iter_count / n_epochs / batch_num * 100, print_loss_avg,print_mt_loss_avg, print_vse_loss_avg)
            print(print_summary)
        
        if iter_count%eval_every == 0:
            #Print the Bleu Score and loss for Dev Dataset
            val_print_loss = 0
            val_print_mt_loss = 0
            val_print_vse_loss = 0
            eval_iters = 0
            val_translations = []

            #Convert model into eval phase
            imagine_model.eval()

            #Compute Val Loss
            for val_x,val_y,val_im,val_x_lengths,val_y_lengths in data_generator_tl_mtv(val_data_index,val_im_feats,batch_size):
                val_loss,val_mt_loss,val_vse_loss = imagine_model(val_x,val_x_lengths,val_y,val_im,teacher_force_ratio,criterion_mt=criterion_mt, criterion_vse=criterion_vse)
                val_print_loss += val_loss.data[0]
                val_print_mt_loss += val_mt_loss.data[0]
                val_print_vse_loss += val_vse_loss.data[0]
                eval_iters += 1

            #Generate translation
            for val_x,val_y,val_im,val_x_lengths,val_y_lengths,val_sorted_index in data_generator_mtv(val_data_index,val_im_feats,eval_batch_size):
                val_translation = imagine_model.beamsearch_decode(val_x,val_x_lengths,val_im,beam_size,max_length=MAX_LENGTH) #Optimize to take in the Image Variables

                #Reorder val_translations and convert them back to words
                val_translation_reorder = translation_reorder_BPE(val_translation,val_sorted_index,t_id2word) 
                val_translations += val_translation_reorder
            
            #Conduct Image Retrieval Evaluation
            #Initialize the lim and ltxt
            val_sample_size = len(val_data_index)
            lim,ltxt = torch.FloatTensor(val_sample_size,shared_embedding_size),torch.FloatTensor(val_sample_size,shared_embedding_size)
            if use_cuda:
                lim = lim.cuda()
                ltxt = ltxt.cuda()

            #Start to generate corresponding im and text features
            for val_x,val_y,val_im,val_x_lengths,index_retrieval in data_generator_tl_mtv_imretrieval(val_data_index,val_im_feats,batch_size):
                index_reorder = [int(x) for x in index_retrieval]
                val_im_vecs, val_txt_vecs = imagine_model.embed_sent_im_test(val_x,val_x_lengths,val_im, max_length=80)
                #Update the Two Matrix
                lim[index_reorder] = val_im_vecs
                ltxt[index_reorder] = val_txt_vecs

            val_r1,val_r5,val_r10,val_medr = im_retrieval_eval.t2i(lim,ltxt)

            #Compute the Average Losses
            val_loss_mean = val_print_loss/eval_iters
            val_mt_loss_mean = val_print_mt_loss/eval_iters
            val_vse_loss_mean = val_print_vse_loss/eval_iters

            #Check the val_mt_loss_mean
            lr_decay_scheduler.step(val_mt_loss_mean)

            #Compute the BLEU Score
            val_bleu = compute_bleu(val_y_ref,val_translations)

            #Compute the METEOR Score
            val_translations_meteor = dict((key,[' '.join(value)]) for key,value in enumerate(val_translations))
            val_meteor = Meteor_Scorer.compute_score(val_y_ref_meteor,val_translations_meteor)
            
            print("dev_loss: {}, dev_mt_loss: {}, dev_vse_loss: {}, dev_bleu: {}, dev_meteor: {}".format(val_loss_mean,val_mt_loss_mean, val_vse_loss_mean,val_bleu[0],val_meteor[0]))
            #Demonstrate the Validation Image Retrieval Accuracy. 
            print("Image Retrieval Accuracy is:")
            print("r1: {}, r5: {}, r10: {}".format(val_r1, val_r5, val_r10))

            #Randomly Pick a sentence and translate it to the target language. 
            sample_source, sample_ref, sample_output = random_sample_display(val_ori_data,val_translations)
            print("An example demo:")
            print("src: {}".format(sample_source))
            print("ref: {}".format(sample_ref))
            print("pred: {}".format(sample_output))
        
            #Save the model when it reaches the best validation loss or best BLEU score
            if val_mt_loss_mean < best_loss:
                torch.save(imagine_model,os.path.join(trained_model_output_path,'nmt_trained_imagine_model_best_loss.pt'))
                #update the best_loss
                best_loss = val_mt_loss_mean

            if val_bleu[0] > best_bleu:
                torch.save(imagine_model,os.path.join(trained_model_output_path,'nmt_trained_imagine_model_best_BLEU.pt'))
                #update the best_bleu score
                best_bleu = val_bleu[0]

            if val_meteor[0] > best_meteor:
              torch.save(imagine_model,os.path.join(trained_model_output_path,'nmt_trained_imagine_model_best_METEOR.pt'))
              #update the best_bleu score
              best_meteor = val_meteor[0]
              early_stop = patience
            else:
              early_stop -= 1


            #Print out the best loss and best BLEU so far
            print("Current Early_Stop Counting: {}".format(early_stop))
            print("Best Loss so far is: {}".format(best_loss))
            print("Best BLEU so far is: {}".format(best_bleu))
            print("Best METEOR so far is: {}".format(best_meteor))
        if iter_count%save_every == 0:
            #Save the model every save_every iterations.
            torch.save(imagine_model,os.path.join(trained_model_output_path,'nmt_trained_imagine_model_{}.pt'.format(iter_count)))
        if early_stop == 0:
            break
        #Update the Iteration
        iter_count += 1
    
    if early_stop == 0:
        break

print("Training is done.")
print("Evalute the Test Result")

#########################Use the best BLEU Model to Evaluate#####################################################
#Load the Best BLEU Model
best_model = torch.load(os.path.join(trained_model_output_path,'nmt_trained_imagine_model_best_BLEU.pt'))
if use_cuda:
    best_model.cuda()

#Convert best_model to eval phase
best_model.eval()

test_translations = []
for test_x,test_y,test_im,test_x_lengths,test_y_lengths,test_sorted_index in data_generator_mtv(test_data_index,test_im_feats,eval_batch_size):
    test_translation = best_model.beamsearch_decode(test_x,test_x_lengths,test_im,beam_size,MAX_LENGTH)

    #Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation,test_sorted_index,t_id2word) 
    test_translations += test_translation_reorder

#Evaluate the Image Retrieval Results
test_sample_size = len(test_data_index)
lim,ltxt = torch.FloatTensor(test_sample_size,shared_embedding_size),torch.FloatTensor(test_sample_size,shared_embedding_size)
if use_cuda:
    lim = lim.cuda()
    ltxt = ltxt.cuda()

#Start to generate corresponding im and text features
for test_x,test_y,test_im,test_x_lengths,index_retrieval in data_generator_tl_mtv_imretrieval(test_data_index,test_im_feats,batch_size):
    index_reorder = [int(x) for x in index_retrieval]
    test_im_vecs, test_txt_vecs = best_model.embed_sent_im_test(test_x,test_x_lengths,test_im, max_length=MAX_LENGTH)
    #Update the Two Matrix
    lim[index_reorder] = test_im_vecs
    ltxt[index_reorder] = test_txt_vecs

test_r1,test_r5,test_r10,test_medr = im_retrieval_eval.t2i(lim,ltxt)
#Generate the test results. 
print("Image Retrieval Accuracy with best_BLEU model is:")
print("r1: {}, r5: {}, r10: {}".format(test_r1, test_r5, test_r10))

#Compute the test bleu score
test_bleu = compute_bleu(test_y_ref,test_translations)
print("test_bleu from the best BLEU model: {}".format(test_bleu[0]))

#Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(trained_model_output_path,'test_2017_prediction_best_BLEU.'+target_language)

with open(test_prediction_path,'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')

#Evalute the final results with nmtpy-coco-metrics
ground_truth_path = os.path.join(data_path,'test'+dataset_suffix+'.'+target_language)

print("Full evaluation results with best BLEU Model:")
#Execute nmtpy-coco-metrics to get the outputs
os.system('nmtpy-coco-metrics {} -l {} -r {}'.format(test_prediction_path,target_language,ground_truth_path))

######################Use the best Loss Model to Evaluate########################################################
#Load the Best Loss Model
best_loss_model = torch.load(os.path.join(trained_model_output_path,'nmt_trained_imagine_model_best_loss.pt'))
if use_cuda:
    best_loss_model.cuda()

#Convert best_model to eval phase
best_loss_model.eval()

test_translations = []
for test_x,test_y,test_im,test_x_lengths,test_y_lengths,test_sorted_index in data_generator_mtv(test_data_index,test_im_feats,eval_batch_size):
    test_translation = best_loss_model.beamsearch_decode(test_x,test_x_lengths,test_im,beam_size,MAX_LENGTH)

    #Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation,test_sorted_index,t_id2word) 
    test_translations += test_translation_reorder

#Evaluate the Image Retrieval Results
test_sample_size = len(test_data_index)
lim,ltxt = torch.FloatTensor(test_sample_size,best_loss_model.shared_embedding_size),torch.FloatTensor(test_sample_size,shared_embedding_size)
if use_cuda:
    lim = lim.cuda()
    ltxt = ltxt.cuda()

#Start to generate corresponding im and text features
for test_x,test_y,test_im,test_x_lengths,index_retrieval in data_generator_tl_mtv_imretrieval(test_data_index,test_im_feats,batch_size):
    index_reorder = [int(x) for x in index_retrieval]
    test_im_vecs, test_txt_vecs = best_loss_model.embed_sent_im_test(test_x,test_x_lengths,test_im, max_length=MAX_LENGTH)
    #Update the Two Matrix
    lim[index_reorder] = test_im_vecs
    ltxt[index_reorder] = test_txt_vecs

test_r1,test_r5,test_r10,test_medr = im_retrieval_eval.t2i(lim,ltxt)
#Generate the test results. 
print("Image Retrieval Accuracy with best_loss model is:")
print("r1: {}, r5: {}, r10: {}".format(test_r1, test_r5, test_r10))

#Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(trained_model_output_path,'test_2017_prediction_best_loss.'+target_language)

with open(test_prediction_path,'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')

#Evalute the final results with nmtpy-coco-metrics
ground_truth_path = os.path.join(data_path,'test'+dataset_suffix+'.'+target_language)

print("Full evaluation results with best Loss model:")
#Execute nmtpy-coco-metrics to get the outputs
os.system('nmtpy-coco-metrics {} -l {} -r {}'.format(test_prediction_path,target_language,ground_truth_path))

###########################Use the best METEOR Model to Evaluate#############################################
#Load the Best Loss Model
best_meteor_model = torch.load(os.path.join(trained_model_output_path,'nmt_trained_imagine_model_best_METEOR.pt'))
if use_cuda:
    best_meteor_model.cuda()

#Convert best_model to eval phase
best_meteor_model.eval()

test_translations = []
for test_x,test_y,test_im,test_x_lengths,test_y_lengths,test_sorted_index in data_generator_mtv(test_data_index,test_im_feats,eval_batch_size):
    test_translation = best_meteor_model.beamsearch_decode(test_x,test_x_lengths,test_im,beam_size,MAX_LENGTH)

    #Reorder val_translations and convert them back to words
    test_translation_reorder = translation_reorder_BPE(test_translation,test_sorted_index,t_id2word) 
    test_translations += test_translation_reorder

#Evaluate the Image Retrieval Results
test_sample_size = len(test_data_index)
lim,ltxt = torch.FloatTensor(test_sample_size,best_meteor_model.shared_embedding_size),torch.FloatTensor(test_sample_size,best_meteor_model.shared_embedding_size)
if use_cuda:
    lim = lim.cuda()
    ltxt = ltxt.cuda()

#Start to generate corresponding im and text features
for test_x,test_y,test_im,test_x_lengths,index_retrieval in data_generator_tl_mtv_imretrieval(test_data_index,test_im_feats,batch_size):
    index_reorder = [int(x) for x in index_retrieval]
    test_im_vecs, test_txt_vecs = best_meteor_model.embed_sent_im_test(test_x,test_x_lengths,test_im, max_length=MAX_LENGTH)
    #Update the Two Matrix
    lim[index_reorder] = test_im_vecs
    ltxt[index_reorder] = test_txt_vecs

test_r1,test_r5,test_r10,test_medr = im_retrieval_eval.t2i(lim,ltxt)
#Generate the test results. 
print("Image Retrieval Accuracy with best_METEOR model is:")
print("r1: {}, r5: {}, r10: {}".format(test_r1, test_r5, test_r10))

#Save the translation prediction to the trained_model_path
test_prediction_path = os.path.join(trained_model_output_path,'test_2017_prediction_best_METEOR.'+target_language)

with open(test_prediction_path,'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')

#Evalute the final results with nmtpy-coco-metrics
ground_truth_path = os.path.join(data_path,'test'+dataset_suffix+'.'+target_language)

print("Full evaluation results with best METEOR model:")
#Execute nmtpy-coco-metrics to get the outputs
os.system('nmtpy-coco-metrics {} -l {} -r {}'.format(test_prediction_path,target_language,ground_truth_path))
