import os
import torch
import argparse
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from ccc import CCC_loss
from utils import get_lr
from aligner import FaceAligner
from breg_next import BReGNeXt, multi_BReGNeXt
from models import MTC_GCN
from dataloading import AffectNet_dataset, Affwild2_dataset, RAF_DB_dataset
from training import train_model_single, eval_model_single, train_model_multi, eval_model_multi

########################################################
# Configuration
########################################################

# Define argument parser
parser = argparse.ArgumentParser(
    description='Train Facial Expression Recognition model using Emotion-GCN')

# Data loading
parser.add_argument('--train_image_dir',
                    default='/root/autodl-tmp/AffectNet/train_set/images',
                    help='path to images of the dataset')
parser.add_argument('--val_image_dir',
                    default='/root/autodl-tmp/AffectNet/val_set/images',
                    help='path to images of the dataset')
parser.add_argument('--data',
                    default='/root/autodl-tmp/AffectNet/data/data_affectnet.pkl',
                    help='path to the pickle file that holds all the information for each sample')
parser.add_argument('--dataset', default='affectnet', type=str,
                    help='Dataset to use (default: affectnet)',
                    choices=['affectnet', 'affwild2', 'raf-db'])
parser.add_argument('--network', default='densenet', type=str,
                    help='Network to use (default: densenet)',
                    choices=['densenet', 'bregnext'])
parser.add_argument('--adj',
                    default='/root/autodl-tmp/AffectNet/data/spearman_affectnet.pkl',
                    help='path to the pickle file that holds the adjacency matrix')
parser.add_argument('--emb',
                    default='/root/autodl-tmp/AffectNet/data/emb.pkl',
                    help='path to the pickle file that holds the word embeddings')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=35, type=int,
                    help='size of each batch (default: 35)')
parser.add_argument('--model', default='emotion_gcn', type=str,
                    help='Model to use (default: emotion_gcn)',
                    choices=['single_task', 'multi_task', 'emotion_gcn'])
# Training
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to train the network (default: 10)')
parser.add_argument('--lambda_multi', default=1, type=float,
                    help='lambda parameter of loss function')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum parameter of SGD (default: 0.9)')
parser.add_argument('--gpu', type=int,
                    help='id of gpu device to use', required=True)
parser.add_argument('--saved_model', type=str,
                    help='name of the saved model', required=True)

# Get arguments from the parser in a dictionary,
args = parser.parse_args()

# Check inputs.
if not os.path.isdir(args.train_image_dir):
    raise FileNotFoundError("Image directory not exists")
if not os.path.isdir(args.val_image_dir):
    raise FileNotFoundError("Image directory not exists")
if not os.path.exists(args.data):
    raise FileNotFoundError("Pickle file not exists")
if args.workers <= 0:
    raise ValueError("Invalid number of workers")
if args.batch_size <= 0:
    raise ValueError("Invalid batch size")
if args.epochs <= 0:
    raise ValueError("Invalid number of epochs")
if args.lr <= 0:
    raise ValueError("Invalid learning rate")
if args.momentum < 0:
    raise ValueError("Invalid momentum value")

# Set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

########################################################
# Save useful parameters of the model
########################################################
output_file = os.path.join('./outputs', args.saved_model)

print("Train Image directory: {}".format(args.train_image_dir))
print("Val Image directory: {}".format(args.val_image_dir))
print("Data file: {}".format(args.data))
print("Dataset name: {}".format(args.dataset))
print("Network name: {}".format(args.network))
print("Adjacency file: {}".format(args.adj))
print("Embeddings file: {}".format(args.emb))
print("Number of workers: {}".format(args.workers))
print("Batch size: {}".format(args.batch_size))
print("Model to use: {}".format(args.model))
print("Number of epochs: {}".format(args.epochs))
print("Lambda {}".format(args.lambda_multi))
print("Learning rate: {}".format(args.lr))
print("Momentum: {}".format(args.momentum))
print("Gpu used: {}".format(args.gpu))

def main():
    ########################################################
    # Define datasets and dataloaders
    ########################################################

    if args.dataset == 'affectnet':
        resized_size = 227
    else:
        resized_size = 112

    if args.network == 'bregnext':
        resized_size = 112

    rotation = 30

    train_transforms = transforms.Compose([
        transforms.Resize((resized_size,resized_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(rotation),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((resized_size,resized_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    aligner = FaceAligner()

    if args.dataset == 'affectnet':
        train_dataset = AffectNet_dataset(root_dir=args.train_image_dir, data_pkl=args.data, emb_pkl=args.emb, aligner=aligner, train=True, transform=train_transforms,
                                        crop_face=True)
        val_dataset = AffectNet_dataset(root_dir=args.val_image_dir, data_pkl=args.data, emb_pkl=args.emb, aligner=aligner, train=False, transform=val_transforms,
                                        crop_face=True)
    elif args.dataset == 'raf-db':
        train_dataset = RAF_DB_dataset(root_dir=args.train_image_dir, data_pkl=args.data, emb_pkl=args.emb,
                                          aligner=aligner, train=True, transform=train_transforms,
                                          crop_face=True)
        val_dataset = RAF_DB_dataset(root_dir=args.val_image_dir, data_pkl=args.data, emb_pkl=args.emb, aligner=aligner,
                                        train=False, transform=val_transforms,
                                        crop_face=True)
    else:
        train_dataset = Affwild2_dataset(data_pkl=args.data, emb_pkl=args.emb, train=True, transform=train_transforms)

        val_dataset = Affwild2_dataset(data_pkl=args.data, emb_pkl=args.emb, train=False, transform=val_transforms)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.workers, pin_memory=True)

    #############################################################################
    # Model Definition (Model, Loss Function, Optimizer)
    #############################################################################


    # model = Emotion_GCN(adj_file=args.adj, input_size=resized_size) # 论文描述的模型
    model = MTC_GCN(adj_file=args.adj, input_size=resized_size)


    # Move the mode weight to cpu or gpu
    model.cuda()
    print(model)

    # Define loss function
    if args.dataset == 'affectnet':
        weights = torch.FloatTensor(
            [3803/74874, 3803/134415, 3803/25459, 3803/14090, 3803/6378, 1, 3803/24882]).cuda()
        criterion_cat = torch.nn.CrossEntropyLoss(weight=weights)
    elif args.dataset == 'raf-db':
        # todo: raf_db的权重
        weights = torch.FloatTensor(
            [281/2524, 281/4772, 281/1982, 281/1290, 1, 281/717, 281/705]).cuda()
        criterion_cat = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion_cat = torch.nn.CrossEntropyLoss()

    criterion_cont = CCC_loss()

    # We optimize only those parameters that are trainable
    params = list(model.parameters())
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    #############################################################################
    # Training Pipeline
    #############################################################################

    # Define lists for train and val loss over each epoch
    train_losses = []
    val_losses = []
    train_cat_losses = []
    val_cat_losses = []
    train_cont_losses = []
    val_cont_losses = []

    # Define variables for early stopping
    max_val_acc = -1
    epochs_no_improve = 0
    n_epochs_stop = 10000

    df = pd.DataFrame(columns=['time', 'epoch', 'batch', 'accuracy', 'loss', 'loss_cat', 'loss_cont',
                               'val_acc', 'val_loss', 'val_loss_cat', 'val_loss_cont',])  # 列名


    if args.dataset == 'affectnet':
        df.to_csv(f'/root/autodl-tmp/result/AffectNet/train_log.csv', index=False)  # 路径可以根据需要更改
    elif args.dataset == 'raf-db':
        df.to_csv(f'/root/autodl-tmp/result/raf-db/train_log.csv', index=False)  # 路径可以根据需要更改
    else:
        df.to_csv(f'/root/autodl-tmp/result/paper2_train_log.csv', index=False)  # 路径可以根据需要更改

    for epoch in range(args.epochs):
        current_lr = get_lr(optimizer)
        print('Current lr: {}'.format(current_lr))

        if args.model == 'single_task':
            train_loss, (y_train_true, y_train_pred) = train_model_single(
                train_dataloader, model, criterion_cat, optimizer)

            val_loss, (y_val_true, y_val_pred) = eval_model_single(
                val_dataloader, model, criterion_cat)
        else:
            train_loss, train_loss_cat, train_loss_cont, (y_train_true, y_train_pred) = train_model_multi(
                args,train_dataloader, model, criterion_cat, criterion_cont, optimizer, gcn=(args.model == 'emotion_gcn'), epoch = epoch)

            val_loss, val_loss_cat, val_loss_cont, (y_val_true, y_val_pred) = eval_model_multi(
                args, val_dataloader, model, criterion_cat, criterion_cont, gcn=(args.model == 'emotion_gcn'))
            
            train_cat_losses.append(train_loss_cat)
            val_cat_losses.append(val_loss_cat)
            train_cont_losses.append(train_loss_cont)
            val_cont_losses.append(val_loss_cont)

        scheduler.step()
        
        # Save losses to the corresponding lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Convert preds and golds in a list.
        y_train_true = np.concatenate(y_train_true, axis=0)
        y_val_true = np.concatenate(y_val_true, axis=0)
        y_train_pred = np.concatenate(y_train_pred, axis=0)
        y_val_pred = np.concatenate(y_val_pred, axis=0)

        # Print metrics for current epoch
        print('Epoch: {}'.format(epoch))
        print("My train loss is : {}".format(train_loss))
        print("My val loss is : {}".format(val_loss))

        if args.model != 'single_task':
            print("My train categorical loss is : {}".format(train_loss_cat))
            print("My val categorical loss is : {}".format(val_loss_cat))
            print("My train continuous loss is : {}".format(train_loss_cont))
            print("My val continuous loss is : {}".format(val_loss_cont))

        print("Accuracy for train: {}".format(accuracy_score(
            y_train_true, y_train_pred)))
        print("Accuracy for val: {}".format(
            accuracy_score(y_val_true, y_val_pred)))

        val_acc = accuracy_score(y_val_true, y_val_pred)

        if val_acc > max_val_acc:
            # Save trained model
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            # Save the model
            torch.save(state, output_file)
            epochs_no_improve = 0
            max_val_acc = val_acc
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            print("Model saved succesfully to {}".format(output_file))
            break

if __name__ == '__main__':
    main()
