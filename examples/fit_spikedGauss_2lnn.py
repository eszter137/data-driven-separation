import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
    
import torch.nn.functional as F
import torchvision
import torch.optim as optim

import random
import numpy.random as rnd
import os

import sys
import subprocess
import time
import argparse

root_dir ="/home/eszekely/"
sys.path.insert(0,root_dir+"/programs/nn/my_functions/")
import define_networks
import general_functions

sys.path.insert(0,root_dir+"/programs/nn/data_nlgp_gp/files/")
import inputs
import tasks


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--num_train",type=int,default=100)
parser.add_argument("--num_test",type=int,default=10000)
parser.add_argument("--l1_width",type=int,default=128)

parser.add_argument("--input_dim",type=int,default=1)
parser.add_argument("--input_length",type=int,default=20)
parser.add_argument("--input_names",nargs='+',type=str,default=["spiked","gp"])
parser.add_argument("--input_gain",type=float,default=3)
parser.add_argument("--num_spike",type=int,default=1)

parser.add_argument("--rand_seed",type=int,default=0)
parser.add_argument("--rand_seed_torch",type=int,default=0)
parser.add_argument("--num_epoch",type=int,default=50)
parser.add_argument('--random_features',choices=("True","False"))
parser.add_argument('--train_bias',choices=("True","False"))
parser.add_argument('--non_linearity',type=str,default="relu")
parser.add_argument("--weight_decay",type=float,default=0.0005)
parser.add_argument("--lr",type=float,default=0.1)


args=parser.parse_args()

num_train=args.num_train
num_test=args.num_test
this_l1_width=args.l1_width

dim=args.input_dim
D=args.input_length
input_names=args.input_names
num_classes=len(input_names)
this_num_classes=len(input_names)
gain = args.input_gain  # how non-linear are the inputs?
num_spike=args.num_spike


rand_seed=args.rand_seed
rand_seed_torch=args.rand_seed_torch
num_epoch=args.num_epoch
use_RF = (args.random_features=='True')
this_train_bias = (args.train_bias=='True')
non_linearity=args.non_linearity
this_wd=args.weight_decay
this_lr=args.lr 

### Generate the data
random.seed(a=rand_seed,version=2)
np.random.seed(seed=rand_seed)
torch.manual_seed(rand_seed_torch)
num_samples = {"train": num_train, "test": num_test}
modes = ["train", "test"]

if num_spike==1:
    str_spike=""
else:
    str_spike="_Nspike"+str(num_spike)

filename_str_dataset=("dim"+str(dim)+"D"+str(D)+"_inputs_"+general_functions.concatenate_list_to_str(input_names)
                      +"_gain"+str(gain)+str_spike)

this_loss="MSELoss"
this_momentum=0.0
useLrCosine=False #True
net=define_networks.TwoLayer(dim*D, this_l1_width)

if not np.isclose(this_wd,0.):
    str_wd="wd"+str(this_wd)
else:
    str_wd=""
if use_RF:
    this_fit_type="2LNN_RF"
else:
    this_fit_type="2LNN"
print("fit type is: ", this_fit_type)

if num_spike>1:
    data_dir="spiked_gaussian_"+str(num_spike)+"spike"
else:
    data_dir="spiked_gaussian"
if useLrCosine:
    tmp_file_name_root=(data_dir+"/"+this_fit_type+"/"+filename_str_dataset
                        +"_w"+str(this_l1_width)+"bias"+str(this_train_bias)[0]+"_ActFunc_"+non_linearity+"_"+this_loss+"LrCosineAnneal"+str_wd+"Momentum"
                        +str(this_momentum)+"Nepoch"+str(num_epoch)
                        +"train_size"+str(num_train)+"test_size"+str(num_test)+"rnd"+str(rand_seed)+"_torchrnd"+str(rand_seed_torch))
else:
    tmp_file_name_root=(data_dir+"/"+this_fit_type+"/"+filename_str_dataset
                        +"_w"+str(this_l1_width)+"bias"+str(this_train_bias)[0]+"_ActFunc_"+non_linearity+"_"+this_loss+"Lr"+str(this_lr)+str_wd+"Momentum"
                        +str(this_momentum)+"Nepoch"+str(num_epoch)
                        +"train_size"+str(num_train)+"test_size"+str(num_test)+"rnd"+str(rand_seed)+"_torchrnd"+str(rand_seed_torch))

my_file_root_data=tmp_file_name_root+"_data"

xs = dict()
ys_target = dict()
ys_prob = dict()

list_spikes=[]
for _i in range(num_spike):
    spike=np.sign(rnd.randn(D))
    list_spikes.append(spike)
    np.savetxt(my_file_root_data+"_spike"+str(_i)+".txt",spike)


distributions = tasks.build_spiked_gaussian_multi_spike(input_names=input_names,D=D,gain=gain,num_spike=num_spike,spikes=list_spikes)
the_task = tasks.Mixture(distributions)
for mode in modes:
    xs[mode],ys_target[mode] =the_task.sample(num_samples[mode])
    tmp_length=(ys_target[mode].size()[0])
    ys_prob[mode] = torch.zeros(tmp_length,this_num_classes)
    for i in range(tmp_length):
        ys_prob[mode][i][int(ys_target[mode][i])]=1.



for mode in modes:
    if np.any(np.isnan(xs[mode].detach().numpy())):
        sys.exit("STOPPING: "+mode+" input contains nans")
    if np.any(np.isnan(ys_prob[mode].detach().numpy())):
        sys.exit("STOPPING: "+mode+" probability list contains nans")



def run_nn_given_inds(net, inds_to_run_on,mode="train"):
    list_target=[]
    list_prediction=[]
    list_loss=[]
    list_probs_all=[]
    list_target_probs_all=[]
    for train_i in inds_to_run_on:
        inputs=xs[mode][train_i]
        list_prob = net(inputs)
        tmp_loss=np.average(((list_prob.detach()).numpy()-ys_prob[mode][train_i].numpy())**2.)

        list_target.append(ys_target[mode][train_i])
        if not np.any(np.isnan(list_prob.detach().numpy())):
            list_prediction.append(list(list_prob).index(np.max(list_prob.detach().numpy())))
        else:
            list_prediction.append(np.NaN)
        list_loss.append(tmp_loss)

    average_loss=np.average(np.array(list_loss))
    N_good_prediction=np.sum(np.array(list_target)==np.array(list_prediction))
    accuracy = ( N_good_prediction ) / float(len(list_target))

    return{"loss": average_loss,
           "accuracy": accuracy}



my_logfile=tmp_file_name_root+".txt"
my_logfile_time=tmp_file_name_root+"_time.txt"
with open(my_logfile, 'w',buffering=1) as f_log:
    f_log.write(f"Epoch Train_loss  Test_loss  "
                +"Train_accuracy  Test_accuracy\n")
with open(my_logfile_time, 'w',buffering=1) as f_time:
    f_time.write("t_after_fit[s]    t_after_test_run[s]\n")

time_begin=time.perf_counter()
net.reset_all_layers()
if this_loss=="MSELoss":
    criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=this_lr, momentum=this_momentum,weight_decay=this_wd)
if useLrCosine:
    optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=10.**(-6),
                                         last_epoch=-1, verbose=False)

print("N =",num_train)
if num_train%this_num_classes==0:
    pass
else:
    print("Warning: number of training points is not dividable by "+str(this_num_classes)+"!!")

train_inds=range(num_train)
running_loss=0.
for epoch in range(num_epoch): 
    shuffled_train_inds=random.sample(train_inds,num_train)
    train_data_full=xs["train"][shuffled_train_inds]
    train_ys_prob_full=ys_prob["train"][shuffled_train_inds]

    for i,data in enumerate(train_data_full):
        inputs=data
        labels=train_ys_prob_full[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

    results_train=run_nn_given_inds(net,inds_to_run_on=train_inds,mode="train")
    time_before_test_run=time.perf_counter()
    results_test=run_nn_given_inds(net,inds_to_run_on=np.arange(num_test), mode="test")
    time_after_test_run=time.perf_counter()

    with open(my_logfile, 'a',buffering=1) as f_log:
        f_log.write(str(epoch)+f"    {results_train['loss']:.6f}    {results_test['loss']:.6f}    "
                    +f"{results_train['accuracy']:.6f}    {results_test['accuracy']:.6f}\n")
    with open(my_logfile_time, 'a',buffering=1) as f_time:
        f_time.write(f"{(time_before_test_run-time_begin):.6f}    {(time_after_test_run-time_begin):.6f}\n")

# save final state of nn.
this_file_nn=tmp_file_name_root+".pth"
torch.save(net.state_dict(), this_file_nn)

print('Finished Training')
