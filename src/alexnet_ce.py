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
from tensorboardX import SummaryWriter

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
NUM_CLASSES = 86  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'ecoset_leuven'
TRAIN_IMG_DIR = 'ecoset_leuven/train'
VALIDATION_IMG_DIR = 'ecoset_leuven/val'
TEST_IMG_DIR = 'ecoset_leuven/test'
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
parser.add_argument('--lr', type=float, default=LR_INIT, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=MOMENTUM, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed ( default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--exp_name', default=False,
                    help='name of the experiment')
parser.add_argument('--alexnet_og_hyperparams', default=True,
                    help='use original alexnet hyper parameters')
parser.add_argument('--lr_decay', type=float, default=LR_DECAY, metavar='LR',
                    help='learning rate decay (default: 0.0005)')
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
parser.add_argument('--weighted_loss', type = bool, default=True)
parser.add_argument('--overfit', type = bool, default=False)
parser.add_argument('--switch_on_lr_decay', type = bool, default=True)
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--run_id', type = str, default = None)

args = parser.parse_args()
# parse command line arguments
if args.alexnet_og_hyperparams:
    args.lr = LR_INIT
else:
    args.lr = 0.001
print(args.weighted_loss, type(args.weighted_loss))

LOG_DIR = '/staging/suresh27/tensorboard/leuven_ecoset' + '/weighted_cross_entropy'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models/{}'.format(args.exp_name)  # model checkpoints
# make checkpoint path directory
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
class_weights_dict = {'towel': 0.8943063698523465,
 'bottle': 0.938717502504878,
 'wasp': 1.1592674841911261,
 'glass': 0.6620412899482667,
 'kettle': 1.926358392762453,
 'llama': 0.9956094723536584,
 'spider': 0.8563806756405692,
 'anvil': 4.752863589031587,
 'iguana': 1.4186923187273857,
 'car': 0.45732922956766586,
 'violin': 0.9019050514262552,
 'submarine': 1.483247648171448,
 'beaver': 1.826089186610724,
 'donkey': 1.2737674418604652,
 'worm': 0.9302795923700026,
 'hammer': 1.1518486883824461,
 'bus': 1.0472411297866207,
 'sieve': 2.279594816104907,
 'cat': 0.3807361524920916,
 'wheelbarrow': 2.6794460751110107,
 'moth': 0.8588680883913925,
 'jar': 1.4148134607130936,
 'elephant': 0.868052880278153,
 'mosquito': 1.1518486883824461,
 'caterpillar': 0.9164808913098321,
 'hedgehog': 2.5397203595377373,
 'bumblebee': 2.5444033104158033,
 'squirrel': 0.8554958020348237,
 'snake': 0.8613699929351876,
 'axe': 0.9670040144934622,
 'sheep': 0.8794867614290444,
 'wolf': 1.0708081184807325,
 'guitar': 0.8711582883094511,
 'ant': 0.1326968678413473,
 'airplane': 0.8509237792490261,
 'knife': 0.8927634647501642,
 'shovel': 1.385920383678109,
 'boat': 3.0350030689490555,
 'toaster': 2.823836416129953,
 'paintbrush': 3.4497868217054264,
 'pig': 0.9847155532936517,
 'monkey': 0.9070429855491918,
 'lion': 0.7034399636442671,
 'beetle': 0.8516239839634873,
 'crowbar': 6.865247406378958,
 'drum': 0.8586899369521908,
 'whale': 0.8633460242015666,
 'dolphin': 0.8709750023241135,
 'cow': 0.8746554375758528,
 'hovercraft': 2.7783518027157794,
 'kangaroo': 1.0522989796762867,
 'screwdriver': 2.430853896680277,
 'bicycle': 0.8544363645090839,
 'alligator': 1.3397230375555054,
 'piano': 0.8761363356712194,
 'horse': 0.6729103033235552,
 'spoon': 1.3822184260589354,
 'wrench': 1.3891759013578897,
 'dog': 0.5103864117921972,
 'tiger': 0.8647888418730962,
 'plate': 1.4591978096744842,
 'zebra': 0.951884154069099,
 'earwig': 2.1371936943967533,
 'tractor': 1.760095317196646,
 'salamander': 1.0109265411590993,
 'deer': 0.8604747840462506,
 'train': 0.8868346585360993,
 'crocodile': 1.3096311882462865,
 'pan': 0.588115383725886,
 'frog': 1.1837987377885364,
 'gecko': 1.9362694976831205,
 'lizard': 0.8560265066266567,
 'truck': 0.8491782945736435,
 'mouse': 0.802587085313399,
 'turtle': 0.8565578700696279,
 'shield': 2.2584529111001155,
 'pliers': 1.43243743461817,
 'bison': 1.175729675105513,
 'bowl': 1.0023593670814799,
 'rabbit': 1.4484759223395771,
 'hamster': 1.3933841083966716,
 'clarinet': 1.8605591847400051,
 'cockroach': 1.3475729772286822,
 'cymbals': 4.625412498375991,
 'chameleon': 1.2932659125418655,
 'helicopter': 1.388709891327243} 

# use wandb api key
wandb.login(key='18a861e71f78135d23eb672c08922edbfcb8d364')
# start a wandb run
id = wandb.util.generate_id()
wandb.init(id = id, resume = "allow", project="semantic-norms-alexnet", entity="siddsuresh97", settings=wandb.Settings(code_dir="vision_robustness_using_semantic_norms/src/alexnet_mse.py"))

config = wandb.config

#name the wandb run
wandb.run.name = args.exp_name




class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 86, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # print the seed value
    # seed = torch.initial_seed()
    seed = args.seed
    torch.manual_seed(seed)
    print('Used seed : {}'.format(seed))

    # tbwriter = SummaryWriter(log_dir=LOG_DIR)
    # print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet(num_classes=args.num_classes).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=args.device_ids)
    print(alexnet)
    print('AlexNet created')

    # create dataset and data loader
    train_dataset = datasets.ImageFolder(args.train_img_dir, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))
    print('Dataset created')
    train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=args.batch_size)
    print('Dataloader created')
    debug_set = torch.utils.data.Subset(train_dataset, list(range(1, args.batch_size*100)))
    debug_data_loader = data.DataLoader(
        debug_set,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=args.batch_size)
    print('Debug dataloader created')
    # add code to load validation data and create validation dataloader
    val_dataset = datasets.ImageFolder(args.validation_img_dir, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))
    print('Validation dataset created')
    val_dataloader = data.DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=args.batch_size)
    print('Validation dataloader created')

    # add code to load test data and create test dataloader
    test_dataset = datasets.ImageFolder(args.test_img_dir, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50616427,0.48602325,0.43117783], std=[0.28661095,0.27966835,0.29607392]),
    ]))
    print('Test dataset created')
    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=args.batch_size)
    print('Test dataloader created')

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