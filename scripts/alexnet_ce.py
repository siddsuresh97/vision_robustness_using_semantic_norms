"""
This implementation has been adapted from the following:
https://github.com/dansuh17/alexnet-pytorch
"""

"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import json
import random
from tensorboardX import SummaryWriter
from src.utils.data_loading import *
from src.models.alexnet import AlexNet

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model parameters
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NDIM = 86
NUM_CLASSES = 86  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'ecoset_leuven'
TRAIN_IMG_DIR = INPUT_ROOT_DIR + '/train'
VALIDATION_IMG_DIR = INPUT_ROOT_DIR + '/val'
TEST_IMG_DIR = INPUT_ROOT_DIR + '/test'
OUTPUT_DIR = 'alexnet_data_out'

#python vision_robustness_using_semantic_norms/src/alexnet.py --exp_name=alexnet_1 --num_classes=86 --weighted_loss=False --alexnet_og_hyperparams=True --device_ids=0,1,2,3 

# parse command line arguments
import argparse
parser = argparse.ArgumentParser(description='PyTorch AlexNet')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=TEST_BATCH_SIZE, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, metavar='N',
                    help='number of epochs to train (default: 90)')
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=MOMENTUM, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed ( default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--exp_name', default=False,
                    help='name of the experiment')
parser.add_argument('--alexnet_og_hyperparams', action='store_true',
                    help='use original alexnet hyper parameters')
parser.add_argument('--lr_decay', type=float, default=LR_DECAY, metavar='LR',
                    help='learning rate decay (default: 0.0005)')
parser.add_argument('--ndim', type=int, default=NDIM, metavar='N',
                    help='ndim(default: 5)')
parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, metavar='N',
                    help='number of classes (default: 86)')
parser.add_argument('--device_ids', nargs='+', type=int, default="0 1 2 3", metavar='N',
                    help='device ids (default: 0, 1, 2, 3)')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, metavar='N',
                    help='output directory (default: alexnet_data_out)')
parser.add_argument('--input_root_dir', type=str, default=INPUT_ROOT_DIR, metavar='N',
                    help='input root directory (default: ecoset_leuven)')
parser.add_argument('--train_img_dir', type=str, default=TRAIN_IMG_DIR, metavar='N',
                    help='train image directory (default: ecoset_leuven/train)')
parser.add_argument('--validation_img_dir', type=str, default=VALIDATION_IMG_DIR, metavar='N',
                    help='validation image directory (default: ecoset_leuven/val)')
parser.add_argument('--test_img_dir', type=str, default=TEST_IMG_DIR, metavar='N',
                    help='test image directory (default: ecoset_leuven/test)')
parser.add_argument('--weighted_loss', action='store_true')
parser.add_argument('--overfit', action='store_true')
parser.add_argument('--switch_on_lr_decay', action='store_true')
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--run_id', type = str, default = None)
parser.add_argument('--eval', type = str, default = 'euclidean')
parser.add_argument('--add_hidden_layers', action='store_true')
parser.add_argument('--triplet_loss', action='store_true')
parser.add_argument('--pre-trained', action='store_true')
parser.add_argument('--wandb_project_name', type=str, metavar='N',
                    help='wandb')

args = parser.parse_args()
# parse command line arguments
if args.alexnet_og_hyperparams == True:
    print('using original alexnet hyperparameters')
    args.lr = LR_INIT
else:
    print('using ADAM')
    args.lr = 0.001

if args.lr != None:
    args.lr = args.lr

LOG_DIR = '/staging/suresh27/tensorboard/leuven_ecoset' + '/weighted_cross_entropy'  # tensorboard logs
CHECKPOINT_DIR = args.output_dir + '/models/{}'.format(args.exp_name)  # model checkpoints
# make checkpoint path directory
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
# read class weights from class_weights.json
with open('vision_robustness_using_semantic_norms/class_weights.json') as f:
    class_weights = json.load(f)
    class_weights_dict = {k: v for k, v in class_weights.items()}

# use wandb api key
wandb.login(key='18a861e71f78135d23eb672c08922edbfcb8d364')
# start a wandb run
id = wandb.util.generate_id()
wandb.init(id = id, resume = "allow", project=args.wandb_project_name, entity="siddsuresh97", settings=wandb.Settings(code_dir="vision_robustness_using_semantic_norms/src/alexnet_ce.py"))

config = wandb.config

#name the wandb run
wandb.run.name = args.exp_name



if __name__ == '__main__':
    # print the seed value
    # seed = torch.initial_seed()
    seed = args.seed
    torch.manual_seed(seed)
    print('Used seed : {}'.format(seed))

    #set random seed
    random.seed(seed)

    # tbwriter = SummaryWriter(log_dir=LOG_DIR)
    # print('TensorboardX summary writer created')

    # create model
    if args.pre_trained:
        alexnet = models.alexnet(pretrained=True)
        print('Loaded pre-trained alexnet model')
        # change the last layer to have 86 output classes
        alexnet.classifier[6] = nn.Sequential(nn.Linear(4096, 2000),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(2000, 500),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(500, 100),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(100, args.ndim))
        print('Changed the last layer to have {} output shape'.format(args.ndim))
        # freeze all the layers except the last layer
        for param in alexnet.parameters():
            param.requires_grad = False
        for param in alexnet.classifier[6].parameters():
            param.requires_grad = True
        # add tanh activation to the last layer
        # alexnet.classifier[6] = nn.Sequential(alexnet.classifier[6], nn.Tanh())
        alexnet = alexnet.to(device)
        print('Freezed all the layers except the last layer')
        
    else:
        alexnet = AlexNet(num_classes=args.ndim, add_hidden_layers=args.add_hidden_layers)
        # add tanh activation to the last layer
        alexnet.classifier[6] = nn.Sequential(alexnet.classifier[6])
        alexnet = alexnet.to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=args.device_ids)
    print(alexnet)
    print('AlexNet created')

    # create dataset and data loader
    train_dataset, val_dataset, test_dataset, debug_dataset = get_train_test_valid_debug_dataset(IMAGE_DIM = IMAGE_DIM, 
                                                                                                                train_img_dir=args.train_img_dir,
                                                                                                                validation_img_dir=args.validation_img_dir,
                                                                                                                test_img_dir=args.test_img_dir, 
                                                                                                                batch_size=args.batch_size)

    train_dataloader, val_dataloader, test_dataloader, debug_data_loader = get_train_test_valid_debug_dataloader(train_dataset, 
                                                                                                                    val_dataset, 
                                                                                                                    test_dataset, 
                                                                                                                    debug_dataset, 
                                                                                                                    batch_size = args.batch_size)

    # use class weights from class weights dictionary by using the class indices and idx2class
    class_weights = [0]*NUM_CLASSES
    for i in train_dataset.classes:
        class_weights[train_dataset.class_to_idx[i]] = class_weights_dict[i]
    class_weights = torch.FloatTensor(class_weights).to(device)
    assert(NUM_CLASSES == len(train_dataset.classes))

    # create optimizer
    # the one that WORKS
    #optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    if args.alexnet_og_hyperparams:
        optimizer = optim.SGD(
            params=alexnet.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.lr_decay)
    else:
        optimizer = optim.Adam(params=alexnet.parameters(), lr=args.lr)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')
    
    # start training!!
    print('Starting training...')
    total_steps = 1
    wandb.watch(alexnet, log='all') 
    start_epoch = 0 
    if args.resume_training:
        print('Resuming training from checkpoint') 
        if args.exp_name == 'alexnet_full_training_one_hot_lr_decay_on':
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(41))
            if os.path.exists(checkpoint_path):
                print('path exists')
                print('Loading checkpoint from {}'.format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                alexnet.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(checkpoint.keys())
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 
                total_steps = checkpoint['total_steps']
                start_epoch = 41 
                wandb_id = checkpoint['wandb_id']
        else:
            for i in range(90, 0, -1):
                checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(i)) 
                if os.path.exists(checkpoint_path):
                    print('Loading checkpoint from {}'.format(checkpoint_path))
                    checkpoint = torch.load(checkpoint_path)
                    alexnet.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(checkpoint.keys())
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 
                    total_steps = checkpoint['total_steps']
                    start_epoch = i 
                    wandb_id = checkpoint['wandb_id']
                    break
        # import ipdb; ipdb.set_trace()
        if args.run_id != None:
            print('using run id given in args')
            wandb.run = wandb.init(resume='must', id=args.run_id)
        else:
            print('using saved run id')
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(start_epoch))
            checkpoint = torch.load(checkpoint_path)
            wandb.run = wandb.init(resume='must', id=checkpoint['wandb_id'])
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(start_epoch, args.epochs):
            if args.overfit:
                train_dataloader = debug_data_loader
            for imgs, classes in train_dataloader:
                alexnet.train()
                imgs, classes = imgs.to(device), classes.to(device)

                # calculate the loss
                output = alexnet(imgs)
                if args.weighted_loss:
                    loss = F.cross_entropy(output, classes, weight=class_weights)
                else:
                    loss = F.cross_entropy(output, classes)
                #loss = F.cross_entropy(output, classes)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log the information and add to tensorboard
                if total_steps % args.log_interval == 0:
                    with torch.no_grad():
                        alexnet.eval()
                        output = alexnet(imgs)
                        # import ipdb; ipdb.set_trace()
                        _, preds = torch.max(output, 1)
                        accuracy = torch.sum(preds == classes)
                        accuracy = accuracy / len(classes)
                        print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                            .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                        # tbwriter.add_scalar('loss', loss.item(), total_steps)
                        # tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)
                        wandb.log({'loss': loss.item(), 'accuracy': accuracy.item()}, step=total_steps)

                        # calculate the validation loss and accuracy
                        val_loss = 0
                        val_accuracy = 0
                        for val_imgs, val_classes in val_dataloader:
                            val_imgs, val_classes = val_imgs.to(device), val_classes.to(device)
                            val_output = alexnet(val_imgs)
                            if args.weighted_loss:
                                val_loss += F.cross_entropy(val_output, val_classes, weight=class_weights)
                            else:
                                val_loss += F.cross_entropy(val_output, val_classes)
                            _, val_preds = torch.max(val_output, 1)
                            val_accuracy += torch.sum(val_preds == val_classes)
                        val_loss = val_loss.item()/len(val_dataloader)
                        val_accuracy = val_accuracy.item()/len(val_dataset)
                        # print('Validation loss: {:.4f} \tValidation accuracy: {:.4f}'
                        #     .format(val_loss, val_accuracy))
                        # tbwriter.add_scalar('val_loss', val_loss, total_steps)
                        # tbwriter.add_scalar('val_accuracy', val_accuracy, total_steps)
                        wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy}, step=total_steps)
                        
                        # calculate the test loss and accuracy
                        test_loss = 0
                        test_accuracy = 0
                        for test_imgs, test_classes in test_dataloader:
                            test_imgs, test_classes = test_imgs.to(device), test_classes.to(device)
                            test_output = alexnet(test_imgs)
                            if args.weighted_loss:    
                                test_loss += F.cross_entropy(test_output, test_classes, weight=class_weights)
                            else:
                                test_loss += F.cross_entropy(test_output, test_classes)
                            _, test_preds = torch.max(test_output, 1)
                            test_accuracy += torch.sum(test_preds == test_classes)
                        test_loss = test_loss.item()/len(test_dataloader)
                        test_accuracy = test_accuracy.item()/len(test_dataset)
                        # print('Test loss: {:.4f} \tTest accuracy: {:.4f}'
                        #     .format(test_loss, test_accuracy))
                        # tbwriter.add_scalar('test_loss', test_loss, total_steps)
                        # tbwriter.add_scalar('test_accuracy', test_accuracy, total_steps)
                        wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})
                        print('Epoch: {} \tStep: {} \tTrain_Loss: {:.4f} \tTrain_Acc: {}\tValidation loss: {:.4f} \tValidation accuracy: {:.4f}\tTest loss: {:.4f} \tTest accuracy: {:.4f}'
                            .format(epoch + 1, total_steps, loss.item(), accuracy.item(), val_loss, val_accuracy, test_loss, test_accuracy))

                # # print out gradient values and parameter average values
                # # if total_steps % args.log_interval == 0:
                # #     with torch.no_grad():
                # #         # print and save the grad of the parameters
                # #         # also print and save parameter values
                # #         print('*' * 10)
                #         for name, parameter in alexnet.named_parameters():
                #             if parameter.grad is not None:
                #                 avg_grad = torch.mean(parameter.grad)
                #                 print('\t{} - grad_avg: {}'.format(name, avg_grad))
                #                 # tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                #                 # tbwriter.add_histogram('grad/{}'.format(name),
                #                 #         parameter.grad.cpu().numpy(), total_steps)
                #                 wandb.log({'grad_avg/{}'.format(name): avg_grad.item()}, step = total_steps)
                #                 wandb.log({'grad/{}'.format(name): wandb.Histogram(parameter.grad.cpu().numpy())}, step = total_steps)
                #             if parameter.data is not None:
                #                 avg_weight = torch.mean(parameter.data)
                #                 print('\t{} - param_avg: {}'.format(name, avg_weight))
                #                 # tbwriter.add_histogram('weight/{}'.format(name),
                #                 #         parameter.data.cpu().numpy(), total_steps)
                #                 # tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
                #                 wandb.log({'weight/{}'.format(name): wandb.Histogram(parameter.data.cpu().numpy())}, step = total_steps)
                #                 wandb.log({'weight_avg/{}'.format(name): avg_weight.item()}, step = total_steps)

                total_steps += 1
            if args.switch_on_lr_decay: 
                lr_scheduler.step()
            # save checkpoints
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
            state = {
                'epoch': epoch + 1,
                'total_steps': total_steps,
                'optimizer': optimizer.state_dict(),
                'model': alexnet.state_dict(),
                'seed': seed,
                'wandb_id': wandb.run.id,
                'lr_scheduler': lr_scheduler.state_dict()
            }
            if not args.overfit:
                torch.save(state, checkpoint_path)