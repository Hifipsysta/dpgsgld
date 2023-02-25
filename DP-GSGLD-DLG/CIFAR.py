import pickle
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import CIFARConvNet
from dpgsgld import DPGSGLD
from dpsgd import DPSGD
from gcsgd import GCSGD
from dpsgld import DPSGLD
from fdpgsgld import FDPGSGLD
from train import train
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import label_to_onehot, cross_entropy_for_onehot


parser = argparse.ArgumentParser(description='DP-GSGLD')

#parser.add_argument('--dataset', type='str', nargs='?', action='store', default='CIFAR10',
                    #help='which dataset is selected as training set and testing set')

parser.add_argument('--batch_size_train', type=int, nargs='?', action = 'store', default=1000, 
                    help='batchsize of train data')

parser.add_argument('--batch_size_test', type=int, nargs='?', action = 'store', default=1000, 
                    help='batchsize of test data')

parser.add_argument('--epochs', type=int, nargs='?', action='store', default=1,
                    help='How many epochs to train. Default: 25.')

parser.add_argument('--learning_rate', type=float, nargs='?', action='store', default=5e-4,
                    help='learning rate for model training.  Default: 1e-3.')

parser.add_argument('--step_size', type=float, nargs='?', action='store', default=1e-2,
                    help='stepsize in Langevin dynamics.  Default: 1e-3.')

# parser.add_argument('--scale', type=float, nargs='?', action='store', default =1,
#                     help='scale parameter for Laplace and Gaussian. Default: 1.')

parser.add_argument('--epsilon', type=float, nargs='?', action='store', default =0.1,
                    help='Privacy budgets for differentially private algorithm. Default: 1.')

parser.add_argument('--delta', type=float, nargs='?', action='store', default =1e-5,
                    help='Failure probability for differentially private algorithm. Default: 1e-5.')

parser.add_argument('--param_threshold', type=float, nargs='?', action='store', default=1,
                    help='threshold of parameter clipping. Default: 1.')

parser.add_argument('--grad_clip', type=float, nargs='?', action='store', default=1,
                    help='threshold of gradient clipping. Default: 1.')

parser.add_argument('--img_index', type=int, nargs='?', action='store', default= 18,
                    help='the index for leaking images on CIFAR.')

parser.add_argument('--iterations', type=int, nargs='?', action='store', default=300,
                    help='Times of iterations for differential model.')

parser.add_argument('--optimizer', type=str, nargs='?', action='store', default='DPGSGLD')

parser.add_argument('--device', type=str, nargs='?', action='store', default='cuda1')
args = parser.parse_args()

print(args)


train_data = torchvision.datasets.CIFAR10('~/data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 [0.49, 0.48, 0.45],[0.25, 0.24, 0.26])
                             ]))

test_data = torchvision.datasets.CIFAR10('~/data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                [0.49, 0.48, 0.45],[0.25, 0.24, 0.26])
                             ]))


training_size = len(train_data)
print('========data size========', training_size)
# print(training_size)
# print(train_data[0][0].shape)
#print(train_data[0])

#training_indices = list(range(training_size))
#print(len(training_indices))
#np.random.shuffle(training_indices)


if args.device == 'cuda0':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif args.device == 'cuda1':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('==========device==========', device)

model = CIFARConvNet() 
model = model.to(device)

#if torch.cuda.device_count() >1:
#    model = nn.DataParallel(model, device_ids=[0,1])


criterion = nn.CrossEntropyLoss()

if args.optimizer == 'DPGSGLD':
    scale = np.sqrt(args.learning_rate * args.param_threshold * args.grad_clip / args.epsilon)
    optimizer = DPGSGLD(params=model.parameters(), grad_clip=args.grad_clip, 
                        scale=scale, lr=args.learning_rate, 
                        batch_size=args.batch_size_train, device=device)

elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)

elif args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

elif args.optimizer == 'DPSGD':
    c_2 = 1
    T = 1
    # q = args.batch_size_train/training_size
    q = 0.5
    sp1 = T * np.log(1/args.delta)
    scale = c_2 * q * np.sqrt(sp1) / args.epsilon

    print("======scale=======", scale)

    optimizer = DPSGD(params=model.parameters(), C = args.grad_clip, 
                        scale=scale, batch_size = args.batch_size_train, 
                        lr=args.learning_rate, device=device)

elif args.optimizer == 'DPSGLD':
    N = len(train_data)
    L = 1
    T = 2*N
    tau = np.floor(np.sqrt(N))

    sp1 = (128 * N * T * pow(L,2)) / (tau * pow(args.epsilon,2))
    sp2 = np.log((2.5 * N * T) / (tau * args.delta))
    sp3 = np.log(2 / args.delta)
    sp4 = pow(args.learning_rate, 2)
    variance = sp1 * sp2 * sp3 * sp4
    variance = min(variance, args.learning_rate)
    scale = np.sqrt(variance)
    # multiplier = N/tau

    print("======scale=======", scale)

    optimizer = DPSGLD(params=model.parameters(), grad_clip=args.grad_clip, 
                        scale=scale, lr=args.learning_rate, 
                        device=device)

elif args.optimizer == 'GC50':
    optimizer = GCSGD(params=model.parameters(), sparsity = 0.5,
                        batch_size = args.batch_size_train, 
                        lr=args.learning_rate, device=device)

elif args.optimizer == 'GC80':
    optimizer = GCSGD(params=model.parameters(), sparsity = 0.8,
                        batch_size = args.batch_size_train, 
                        lr=args.learning_rate, device=device)

elif args.optimizer == 'FDPGSGLD':
    scale = np.sqrt(args.learning_rate * args.param_threshold * args.grad_clip / args.epsilon)
    optimizer = FDPGSGLD(params=model.parameters(), grad_clip=args.grad_clip, 
                            scale=scale, lr=args.learning_rate, 
                            batch_size=args.batch_size_train, 
                            n_epochs=args.epochs, device=device)



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
    argsopt=args.optimizer
    )

print('Total Time Consume: ', np.sum(time_consume_history))



def test(model, device, test_data, batch_size_train):
    test_loss = 0                           
    correct = 0  

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size_train, 
        shuffle=True
        )

    model.eval()                          
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)            
            test_loss += criterion(output, target).item() 
            pred = output.max(1, keepdim=True)[1]       
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test(model = model, device = device, test_data = test_data, batch_size_train=args.batch_size_test)




gt_data = train_data[args.img_index][0]

#origin_img=gt_data.permute(1,2,0)

#plt.figure()
#plt.imshow(origin_img)
#plt.axis('off')
#plt.savefig('assets/origin_img_CIFAR.png', bbox_inches='tight')
#print(gt_data.shape)



img_index = args.img_index
gt_data = (train_data[img_index][0]).to(device)
print(gt_data.shape)

gt_data = gt_data.view(1, *gt_data.size())
print(gt_data.shape)

gt_label = train_data[img_index][1]
print(gt_label)
gt_label = torch.tensor(gt_label).long().to(device)
print(gt_label)

gt_label = gt_label.view(1, )
print(gt_label)
gt_onehot_label = label_to_onehot(gt_label, num_classes = 10)
print(gt_onehot_label)

img_data = gt_data[0]
torchvision.utils.save_image(img_data, 'assets/original_CIFAR_'f'{args.img_index}.png') 


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
tt = transforms.ToPILImage()

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
                                        'assets/CIFAR10_'f'{args.optimizer}_{args.epsilon}_{args.img_index}.png')
        current_loss = closure()
        print(iters+1, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))




def save_as_pickle(save_dir, save_data):
    if type(save_dir) == str and type(save_data)==str:
        save_path = save_dir + '/' + save_data  +'.pickle'
        save_file = open(save_path, 'wb')
        pickle.dump(eval(save_data), save_file)
        save_file.close()
    else:
        print('Please enter the correct variable type')
#save_as_pickle('assets', 'history')





#plt.figure()
#plt.imshow(history[-1])
#plt.axis('off')
#plt.savefig('assets/CIFAR10_'f'{args.optimizer}_{args.img_index}.png', bbox_inches='tight')
#torchvision.utils.save_image(f'{CIFAR10}_{args.optimizer}_{args.img_index}.png')

'''
plt.figure(figsize=(12, 4))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % ((i+1) * 10))
    plt.axis('off')

plt.savefig('assets/disclose.png', bbox_inches='tight')
'''



#save_as_pickle('evaluation_dir', 'train_loss_history')
#save_as_pickle('evaluation_dir', 'time_consume_history')
#save_as_pickle('evaluation_dir', 'gradient_norm_history')
#save_as_pickle('evaluation_dir', 'train_accuracy_history')
