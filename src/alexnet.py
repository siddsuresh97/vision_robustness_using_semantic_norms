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
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'ecoset_leuven'
TRAIN_IMG_DIR = 'ecoset_leuven/train'
VAL_IMG_DIR = 'ecoset_leuven/val'
TEST_IMG_DIR = 'ecoset_leuven/test'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = '/staging/suresh27/tensorboard/leuven_ecoset' + '/weighted_cross_entropy'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# start a wandb run
# import wandb
# wandb.init(project="leuven_ecoset", entity="suresh27", config={
#     "epochs": NUM_EPOCHS,
#     "batch_size": BATCH_SIZE,
#     "momentum": MOMENTUM,
#     "lr_decay": LR_DECAY,
#     "lr_init": LR_INIT,
#     "image_dim": IMAGE_DIM,
#     "num_classes": NUM_CLASSES,
#     "device_ids": DEVICE_IDS,
#     "input_root_dir": INPUT_ROOT_DIR,
#     "train_img_dir": TRAIN_IMG_DIR,
#     "output_dir": OUTPUT_DIR,
#     "log_dir": LOG_DIR,
#     "checkpoint_dir": CHECKPOINT_DIR,
# })
# # name the run
wandb.run.name = "alexnet_leuven_ecoset"

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
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
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('AlexNet created')

    # create dataset and data loader
    train_dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')
    train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    # add code to load validation data and create validation dataloader
    val_dataset = datasets.ImageFolder(VALIDATION_IMG_DIR, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Validation dataset created')
    val_dataloader = data.DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Validation dataloader created')

    # add code to load test data and create test dataloader
    test_dataset = datasets.ImageFolder(TEST_IMG_DIR, transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Test dataset created')
    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Test dataloader created')



    # create weights for each class to account for class imbalance. This is done by
    # counting the number of samples in each class and dividing by the total number of samples
    # to get the class weights. These weights are then used in the loss function.
    class_weights = [0] * NUM_CLASSES
    for _, label in dataset:
        class_weights[label] += 1
    total_samples = sum(class_weights)
    class_weights = [total_samples / c for c in class_weights]
    print('Class weights: {}'.format(class_weights))


    # create optimizer
    # the one that WORKS
    #optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    optimizer = optim.SGD(
        params=alexnet.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=LR_DECAY)
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')
    
    # start training!!
    print('Starting training...')
    total_steps = 1
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(NUM_EPOCHS):
            for imgs, classes in dataloader:
                
                imgs, classes = imgs.to(device), classes.to(device)

                # calculate the loss
                output = alexnet(imgs)
                loss = F.cross_entropy(output, classes, weight=class_weights)
                #loss = F.cross_entropy(output, classes)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # log the information and add to tensorboard
                if total_steps % 10 == 0:
                    with torch.no_grad():
                        _, preds = torch.max(output, 1)
                        accuracy = torch.sum(preds == classes)
                        print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                            .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                        tbwriter.add_scalar('loss', loss.item(), total_steps)
                        tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

                        # calculate the validation loss and accuracy
                        val_loss = 0
                        val_accuracy = 0
                        for val_imgs, val_classes in val_dataloader:
                            val_imgs, val_classes = val_imgs.to(device), val_classes.to(device)
                            val_output = alexnet(val_imgs)
                            val_loss += F.cross_entropy(val_output, val_classes, weight=class_weights)
                            _, val_preds = torch.max(val_output, 1)
                            val_accuracy += torch.sum(val_preds == val_classes)
                        val_loss /= len(val_dataloader)
                        val_accuracy /= len(val_dataset)
                        # print('Validation loss: {:.4f} \tValidation accuracy: {:.4f}'
                        #     .format(val_loss, val_accuracy))
                        tbwriter.add_scalar('val_loss', val_loss, total_steps)
                        tbwriter.add_scalar('val_accuracy', val_accuracy, total_steps)
                        
                        # calculate the test loss and accuracy
                        test_loss = 0
                        test_accuracy = 0
                        for test_imgs, test_classes in test_dataloader:
                            test_imgs, test_classes = test_imgs.to(device), test_classes.to(device)
                            test_output = alexnet(test_imgs)
                            test_loss += F.cross_entropy(test_output, test_classes, weight=class_weights)
                            _, test_preds = torch.max(test_output, 1)
                            test_accuracy += torch.sum(test_preds == test_classes)
                        test_loss /= len(test_dataloader)
                        test_accuracy /= len(test_dataset)
                        # print('Test loss: {:.4f} \tTest accuracy: {:.4f}'
                        #     .format(test_loss, test_accuracy))
                        tbwriter.add_scalar('test_loss', test_loss, total_steps)
                        tbwriter.add_scalar('test_accuracy', test_accuracy, total_steps)
                        print('Epoch: {} \tStep: {} \tTrain_Loss: {:.4f} \tTrain_Acc: {}\tValidation loss: {:.4f} \tValidation accuracy: {:.4f}\tTest loss: {:.4f} \tTest accuracy: {:.4f}'
                            .format(epoch + 1, total_steps, loss.item(), accuracy.item(), val_loss, val_accuracy, test_loss, test_accuracy))

                # print out gradient values and parameter average values
                if total_steps % 100 == 0:
                    with torch.no_grad():
                        # print and save the grad of the parameters
                        # also print and save parameter values
                        print('*' * 10)
                        for name, parameter in alexnet.named_parameters():
                            if parameter.grad is not None:
                                avg_grad = torch.mean(parameter.grad)
                                print('\t{} - grad_avg: {}'.format(name, avg_grad))
                                tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                                tbwriter.add_histogram('grad/{}'.format(name),
                                        parameter.grad.cpu().numpy(), total_steps)
                            if parameter.data is not None:
                                avg_weight = torch.mean(parameter.data)
                                print('\t{} - param_avg: {}'.format(name, avg_weight))
                                tbwriter.add_histogram('weight/{}'.format(name),
                                        parameter.data.cpu().numpy(), total_steps)
                                tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

                total_steps += 1

            # save checkpoints
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
            state = {
                'epoch': epoch,
                'total_steps': total_steps,
                'optimizer': optimizer.state_dict(),
                'model': alexnet.state_dict(),
                'seed': seed,
            }
            torch.save(state, checkpoint_path)