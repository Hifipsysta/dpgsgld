import os
import pickle
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model import LFWConvNet
from resnet18 import ResNet18
from dpgsgld import DPGSGLD
from dpsgld import DPSGLD
from dpsgd import DPSGD
from gcsgd import GCSGD
from data import Make_data
from train import train
import matplotlib.pyplot as plt
from utils import label_to_onehot, cross_entropy_for_onehot


parser = argparse.ArgumentParser(description='DP-GSGLD')

parser.add_argument('--batch_size_train', type=int, nargs='?', action = 'store', default=16, 
                    help='batchsize of train data')

parser.add_argument('--batch_size_test', type=int, nargs='?', action = 'store', default=16, 
                    help='batchsize of test data')

parser.add_argument('--image_threshold', type=int, nargs='?', action = 'store', default=30)

parser.add_argument('--epochs', type=int, nargs='?', action='store', default=1,
                    help='How many epochs to train. Default: 25.')

parser.add_argument('--learning_rate', type=float, nargs='?', action='store', default=1e-2,
                    help='learning rate for model training.  Default: 1e-3.')

parser.add_argument('--scale', type=float, nargs='?', action='store', default =1,
                    help='scale parameter for Laplace and Gaussian. Default: 1.')

parser.add_argument('--param_threshold', type=float, nargs='?', action='store', default=1,
                    help='threshold of parameter clipping. Default: 1.')

parser.add_argument('--grad_clip', type=float, nargs='?', action='store', default=1,
                    help='threshold of gradient clipping. Default: 1.')

parser.add_argument('--img_index', type=int, nargs='?', action='store', default=100,
                    help='the index for leaking images on CIFAR.')

parser.add_argument('--iterations', type=int, nargs='?', action='store', default=300,
                    help='Times of iterations for differential model.')

parser.add_argument('--optimizer', type=str, nargs='?', action='store', default='DPGSGLD')

parser.add_argument('--device', type=str, nargs='?', action='store', default='cuda1')

parser.add_argument('--save_gradient', type=bool, nargs='?', action='store', default=False)

args = parser.parse_args()


train_data = torchvision.datasets.LFWPeople('LFWPeople', download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                               mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                             ]))

print('=======Total sample size of dataset=======',len(train_data))

selected_people = []
dir_path = os.listdir(r'LFWPeople/lfw-py/lfw_funneled/')
for item in dir_path:
    #if item not in ['.DS_Store', 'pairs_08.txt','pairs_09.txt', 'pairs.txt']:
    dir_list = os.path.join(r'LFWPeople/lfw-py/lfw_funneled/', str(item))
    if os.path.isdir(dir_list):
        if len(os.listdir(dir_list)) >= args.image_threshold:
            #print(dir_list, len(os.listdir(dir_list)))
            selected_people.append(dir_list)


data_path_list = []
label_idx_list = []
for i in range(len(selected_people)):
    for j in range(len(os.listdir(selected_people[i]))):
        img_name = os.listdir(selected_people[i])[j]
        full_path = os.path.join(selected_people[i], img_name)
        
        data_path_list.append(full_path)
        label_idx_list.append(i)


train_data = Make_data(data_path_list, label_idx_list)
#train_loader = DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, drop_last=True)
#print(train_data[0][0])


#print(len(train_loader))

if args.device == 'cuda0':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif args.device == 'cuda1':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


model = ResNet18(num_classes = len(selected_people)).to(device)
#model = LFWConvNet(nClasses = len(selected_people)).to(device)

#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model, device_ids=[0,1])


criterion = nn.CrossEntropyLoss()

if args.optimizer == 'DPGSGLD':
    optimizer = DPGSGLD(params=model.parameters(), grad_clip=args.grad_clip, 
        scale=args.scale, lr=args.learning_rate, 
        batch_size=args.batch_size_train, device=device)

elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)

elif args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

elif args.optimizer == 'DPSGD':
    optimizer = DPSGD(params=model.parameters(), C = args.grad_clip, 
                    scale=args.scale, batch_size = args.batch_size_train, 
                    lr=args.learning_rate, device=device)

elif args.optimizer == 'DPSGLD':
    optimizer = DPSGLD(params=model.parameters(), grad_clip=args.grad_clip, 
                        scale=args.scale, lr=args.learning_rate, 
                        batch_size=args.batch_size_train, device=device)

elif args.optimizer == 'GC50':
    optimizer = GCSGD(params=model.parameters(), sparsity = 0.5, 
                    scale=args.scale, batch_size = args.batch_size_train, 
                    lr=args.learning_rate, device=device)

elif args.optimizer == 'GC80':
    optimizer = GCSGD(params=model.parameters(), sparsity = 0.8, 
                    scale=args.scale, batch_size = args.batch_size_train, 
                    lr=args.learning_rate, device=device)




#for batch_idx,(feature, target) in enumerate(train_loader):
#    print(target)


param_history = []
gradient_norm_history = []
train_loss_history = []
time_consume_history = []
train_accuracy_history = []



train_loss_history, time_consume_history, \
param_history, gradient_norm_history, train_accuracy_history= train(
    model = model, n_epochs = args.epochs, device = device, 
    criterion = criterion, optimizer = optimizer, 
    train_data = train_data, batch_size_train=args.batch_size_train,
    train_accuracy_history = train_accuracy_history,
    param_history = param_history, 
    param_threshold = args.param_threshold,
    gradient_norm_history = gradient_norm_history, 
    train_loss_history = train_loss_history, 
    time_consume_history = time_consume_history,
    argsopt = args.optimizer
    )


print('Total Time Consume: ', np.sum(time_consume_history))



if args.save_gradient == True:
    def save_as_pickle(save_dir, save_data):
        if type(save_dir) == str and type(save_data)==str:
            save_path = save_dir + '/' + save_data + '_' + args.optimizer + '.pickle'
            save_file = open(save_path, 'wb')
            pickle.dump(eval(save_data), save_file)
            save_file.close()
            print('Gradient saved.')
        else:
            print('Please enter the correct variable type')

    save_as_pickle('assets', 'gradient_norm_history')
else:
    pass










gt_data = train_data[args.img_index][0]

#origin_img=gt_data.permute(1,2,0)



gt_data = gt_data.view(1, *gt_data.size()).to(device)

print(gt_data.shape)

gt_label = train_data[args.img_index][1]
gt_label = torch.tensor(gt_label).long().to(device)
print(gt_label)

gt_label = gt_label.view(1, )
print(gt_label)
gt_onehot_label = label_to_onehot(target = gt_label, num_classes = len(selected_people))
print(gt_onehot_label)

img_data = gt_data[0]
torchvision.utils.save_image(img_data, 'assets/original_LFW_'f'{args.img_index}.png') 






tt = torchvision.transforms.ToPILImage()


# compute original gradient 
pred = model(gt_data)
print(pred)

criterion = cross_entropy_for_onehot
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, model.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))


# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)


#img_dummy = dummy_data[0]
#torchvision.utils.save_image(img_dummy, 'assets/dummy_data.png') 


optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


history = []


for iters in range(args.iterations):
    def closure():
        optimizer.zero_grad()

        dummy_pred = model(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if (iters+1) % 10 == 0: 
        if iters == args.iterations-1:
            torchvision.utils.save_image(dummy_data[0],
                                        'assets/LFW_'f'{args.optimizer}_{args.scale}_{args.grad_clip}_{args.iterations}-{args.img_index}.png')

        current_loss = closure()
        print(iters+1, "%.4f" % current_loss.item())



        #saved_img = tt(dummy_data[0].cpu())
        #history.append(saved_img)


        #plt.figure()
        #plt.imshow(saved_img)
        #plt.title('iter=%d'% (iters))
        #plt.axis('off')
        #plt.savefig('assets/'+str(iters)+'.png', bbox_inches='tight')






'''
plt.figure(figsize=(12, 4))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.savefig('assets/disclose.png', bbox_inches='tight')


save_as_pickle('assets', 'history')
'''