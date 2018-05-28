'''
Implement the beam search version of nmt_monomodal_beam.py 
'''

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import subprocess

from preprocessing import *
from bleu import *
from machine_translation_vision.models import NMT_Seq2Seq_Beam, LIUMCVC_Seq2Seq_Beam, NMT_Seq2Seq_Beam_V2
#from machine_translation_vision.models import LIUMCVC_Seq2Seq
from train import *

import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

use_cuda = torch.cuda.is_available()
print("Whether GPU is available: {}".format(use_cuda))

#The token index for the start of the sentence
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 80 #We will abandon any sentence that is longer than this length

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

def save_plot(points,save_path,y_label):
    plt.plot(points)
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
#Load the Dataset
data_path = '/home/zmykevin/machine_translation_vision/dataset/Multi30K_DE_BPE_Kevin'
trained_model_output_path = '/home/zmykevin/machine_translation_vision/code/mtv_trained_model/WMT17/nmt_DE_20_4'
source_language = 'en'
target_language = 'de'
BPE_dataset_suffix = '.norm.tok.lc.10000bpe'
dataset_suffix = '.norm.tok.lc'

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



#Creating List of pairs in the format of [[en_1,de_1], [en_2, de_2], ....[en_3, de_3]] for BPE data
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


#Network Structure
embedding_size = 256
hidden_size = 512
n_layers = 1
tied_emb = True

#Dropout
dropout_rnn = 0.0
dropout_emb = 0.3
dropout_ctx = 0.5
dropout_out = 0.5

#Training Setting
batch_size = 32
eval_batch_size = 16
batch_num = math.floor(len(train_data_index)/batch_size)
learning_rate = 0.0004
weight_decay = 0.00001
beam_size = 12
n_epochs = 100
print_every = 100
eval_every = 1000
save_every = 10000
teacher_forcing_ratio = 0.8
clip = 1.0
patience = 10

#Initialize models
input_size = len(s_word2id)+1
output_size = len(t_word2id)+1
baseline_model = NMT_Seq2Seq_Beam_V2(input_size,\
                             output_size,\
                             embedding_size,\
                             embedding_size,\
                             hidden_size,\
                             n_layers=n_layers,\
                             dropout_ctx=dropout_ctx,\
                             dropout_emb=dropout_emb,\
                             dropout_out=dropout_out,\
                             dropout_rnn=dropout_rnn,\
                             tied_emb=tied_emb)


#Move models to GPU
if use_cuda:
    print("Move Models to GPU")
    baseline_model.cuda()


#Initialize optimization and criterion    
#optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

weight_group = { 'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, baseline_model.named_parameters())) if 'bias' not in n],
                 'weight_decay': weight_decay,
}

bias_group = { 'params': [p for n,p in list(filter(lambda p: p[1].requires_grad, baseline_model.named_parameters())) if 'bias' in n],
    
}

param_groups = [weight_group,bias_group]
optimizer = optim.Adam(param_groups,lr=learning_rate) #Optimize the parameters

#Including the mask to ignore the <pad> to compute the loss

vocab_mask = torch.ones(output_size)
vocab_mask[0] = 0
if use_cuda:
    vocab_mask = vocab_mask.cuda()

criterion = nn.NLLLoss(weight=vocab_mask,reduce=False)
#criterion = nn.NLLLoss(weight=vocab_mask)

#Define a learning rate optimizer
lr_decay_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=10)

#keep track of time elapsed and running averages
start = time.time()
print_loss_total = 0 #Reset every print_every

################################Print the configuration settings#################################
print("Configurations:")
print('\n')
print("######## Network Structure #########")
print("embedding_size: {}".format(embedding_size))
print("hidden_size: {}".format(hidden_size))
print("n_layers: {}".format(n_layers))
print("tied_emb: {}".format(tied_emb))
print('\n')
print("####### Dropout #######")
print("dropout_rnn: {}".format(dropout_rnn))
print("dropout_emb: {}".format(dropout_emb))
print("dropout_ctx: {}".format(dropout_ctx))
print("dropout_out: {}".format(dropout_out))
print('\n')
print("####### Training Setting #######")
print("batch_size: {}".format(batch_size))
print("eval_batch_size: {}".format(eval_batch_size))
print("learning_rate: {}".format(learning_rate))
print("weight_decay: {}".format(weight_decay))
print("beam_size: {}".format(beam_size))
print("n_epochs: {}".format(n_epochs))
print("print_every: {}".format(print_every))
print("eval_every: {}".format(eval_every))
print("save_every: {}".format(save_every))
print("teacher_force_ratio: {}".format(teacher_forcing_ratio))
print("clip: {}".format(clip))
print("input_size: {}".format(input_size))
print("output_size: {}".format(output_size))
print('\n')

#Begin Training
print("Begin Training")

#Initialize some parameters fro this training process
iter_count = 0
best_bleu = 0
best_loss = 1000
early_stop=patience
for epoch in range(1,n_epochs + 1):
    for batch_x,batch_y,batch_x_lengths,batch_y_lengths in data_generator_tl(train_data_index,batch_size):
        input_variable = batch_x  #B*W_x
        target_variable = batch_y #B*W_y
        #Run the train function
        #loss = train_nmt(input_variable, target_variable, batch_x_lengths,batch_y_lengths,baseline_model,criterion,optimizer,teacher_force_ratio=teacher_forcing_ratio)
        loss = train_nmt(input_variable, target_variable, batch_x_lengths,baseline_model,criterion,optimizer,teacher_force_ratio=teacher_forcing_ratio)
        print_loss_total += loss
    
        if iter_count == 0: 
            iter_count += 1
            continue
        
        if iter_count%print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, iter_count / n_epochs / batch_num), iter_count, iter_count / n_epochs / batch_num * 100, print_loss_avg)
            print(print_summary)
        
        if iter_count%eval_every == 0:
            #Print the Bleu Score and loss for Dev Dataset
            val_losses = 0
            eval_iters = 0
            val_translations = []
            
            #Convert baseline_model into eval phase
            baseline_model.eval()
            
            #Compute loss
            for val_x,val_y,val_x_lengths,val_y_lengths in data_generator_tl(val_data_index,eval_batch_size):
                val_loss = baseline_model(val_x,val_x_lengths,val_y,teacher_force_ratio=teacher_forcing_ratio,criterion=criterion)
                val_losses += val_loss.data[0]
                eval_iters += 1
            

            #Generate translation
            for val_x,val_y,val_x_lengths,val_y_lengths,val_sorted_index in data_generator(val_data_index,eval_batch_size):
                val_translation = baseline_model.beamsearch_decode(val_x,val_x_lengths,beam_size,max_length=MAX_LENGTH)
                
                #Reorder val_translations and convert them back to words
                val_translation_reorder = translation_reorder_BPE(val_translation,val_sorted_index,t_id2word) 
                val_translations += val_translation_reorder
                
            val_loss_mean = val_losses/eval_iters
            #Check the Val_mt_loss_mean
            lr_decay_scheduler.step(val_loss_mean)

            val_bleu = compute_bleu(val_y_ref,val_translations)
            print("dev_loss: {}, dev_bleu: {}".format(val_loss_mean,val_bleu[0]))

            #Randomly Pick a sentence and translate it to the target language. 
            sample_source, sample_ref, sample_output = random_sample_display(val_ori_data,val_translations)
            print("An example demo:")
            print("src: {}".format(sample_source))
            print("ref: {}".format(sample_ref))
            print("pred: {}".format(sample_output))

            #Save the model when it reaches the best validation loss or best BLEU score
            if val_loss_mean < best_loss:
                torch.save(baseline_model,os.path.join(trained_model_output_path,'nmt_wmt_baseline_trained_model_best_loss.pt'))
                #update the best_loss
                best_loss = val_loss_mean

            if val_bleu[0] > best_bleu:
                torch.save(baseline_model,os.path.join(trained_model_output_path,'nmt_wmt_baseline_trained_model_best_BLEU.pt'))
                #update the best_bleu score
                best_bleu = val_bleu[0]
                early_stop = patience
            else:
                early_stop -= 1

            #Print out the best loss and best BLEU so far
            print("Current Early_Stop Counting: {}".format(early_stop))
            print("Best Loss so far is: {}".format(best_loss))
            print("Best BLEU so far is: {}".format(best_bleu))

            #Schedule a learning rate decay
            #lr_decay_scheduler.step(val_loss_mean)
      
        if iter_count%save_every == 0:
            #Save the model every save_every iterations.
            torch.save(baseline_model,os.path.join(trained_model_output_path,'nmt_wmt_baseline_trained_model_{}.pt'.format(iter_count)))
        if early_stop == 0:
            break
        iter_count += 1
    if early_stop==0:
        break
print("Training is done.")
#Save the Model
print("Evalute the Test Result")

#########################Use the best BLEU Model to Evaluate#####################################################
#Load the Best BLEU Model
best_model = torch.load(os.path.join(trained_model_output_path,'nmt_wmt_baseline_trained_model_best_BLEU.pt'))
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
best_loss_model = torch.load(os.path.join(trained_model_output_path,'nmt_wmt_baseline_trained_model_best_loss.pt'))
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
test_prediction_path = os.path.join(trained_model_output_path,'test_2017_prediction_best_loss.'+target_language)

with open(test_prediction_path,'w') as f:
    for x in test_translations:
        f.write(' '.join(x)+'\n')

#Evalute the final results with nmtpy-coco-metrics
ground_truth_path = os.path.join(data_path,'test'+dataset_suffix+'.'+target_language)

print("Full evaluation results with best Loss model:")
#Execute nmtpy-coco-metrics to get the outputs
os.system('nmtpy-coco-metrics {} -l {} -r {}'.format(test_prediction_path,target_language,ground_truth_path))




