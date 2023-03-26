import argparse
from datetime import datetime

import torch
import os

from thop import profile, clever_format
from torchstat import stat

from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from aligner import FaceAligner
from ccc import CCC_loss
from dataloading import AffectNet_dataset, RAF_DB_dataset
from models import MTC_GCN
from training import eval_model_multi
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    mean_squared_error, mean_absolute_error


def cat_result(y_true, y_pred):


    print("Accuracy for val: {}".format(
        accuracy_score(y_true, y_pred)))

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(7))
    # 打印混淆矩阵
    print("Confusion matrix:\n", cm)

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 打印准确率
    print("Accuracy:", accuracy)

    # 计算精确率
    precision = precision_score(y_true, y_pred, average='macro')
    # 打印精确率
    print("Precision:", precision)

    # 计算召回率
    recall = recall_score(y_true, y_pred, average='macro')
    # 打印召回率
    print("Recall:", recall)

    # 计算f1值
    f1 = f1_score(y_true, y_pred, average='macro')
    # 打印f1值
    print("F1 score:", f1)



    for i in range(7):
        subset_true = []
        subset_pred = []
        for j in range(len(y_true)):
            if y_true[j] == i:
                subset_true = np.append(subset_true, 1)
            else:
                subset_true = np.append(subset_true, 0)
            if y_pred[j] == i:
                subset_pred = np.append(subset_pred, 1)
            else:
                subset_pred = np.append(subset_pred, 0)


            # if y_true[j] == i:
            #     subset_true = np.append(subset_true, 1)
            #     if y_pred[j] == i:
            #         subset_pred = np.append(subset_pred, 1)
            #     else:
            #         subset_pred = np.append(subset_pred, 0)

        print('=========================================')
        print("class: ", i)
        print("Accuracy:", accuracy_score(subset_true, subset_pred))
        print("Precision:", precision_score(subset_true, subset_pred))
        print("Recall:", recall_score(subset_true, subset_pred))
        print("F1 score:", f1_score(subset_true, subset_pred))



def test_model_multi(args, dataloader, model, criterion_cat, criterion_cont, gcn=False):
    model.eval()
    running_loss = 0.0
    running_loss_cat = 0.0
    running_loss_cont = 0.0
    y_pred = []
    y_true = []

    yv_cont_true = []
    yv_cont_pred = []

    ya_cont_true = []
    ya_cont_pred = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            inputs, labels, labels_cont, inp = batch


            model.to(device)
            inp = inp.to(device)
            myinput = inputs.to(device)
            flops, params = profile(model.to(device), inputs=(myinput, inp))
            flops, params = clever_format([flops, params], "%.3f")
            print(flops, params)

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_cont = labels_cont.to(device)
            inp = inp.to(device)
            # stat(model, (3, 224, 224), inp)
            if gcn:
                outputs_cat, outputs_cont = model(inputs, inp)
            else:
                outputs_cat, outputs_cont = model(inputs)

            loss_cat = criterion_cat(outputs_cat, labels)
            loss_cont = criterion_cont(
                outputs_cont.double(), labels_cont.double())
            _, preds = torch.max(outputs_cat, 1)
            loss = loss_cat + loss_cont

            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())



            yv_cont_pred.append(outputs_cont[:,0].cpu().numpy())
            yv_cont_true.append(labels_cont[:,0].cpu().numpy())

            ya_cont_pred.append(outputs_cont[:,1].cpu().numpy())
            ya_cont_true.append(labels_cont[:,1].cpu().numpy())

            running_loss += loss.data.item()
            running_loss_cat += loss_cat.data.item()
            running_loss_cont += loss_cont.data.item()
            y_train_true = np.concatenate(y_true, axis=0)
            y_train_pred = np.concatenate(y_pred, axis=0)


    return running_loss / index, running_loss_cat / index, running_loss_cont / index,\
           (y_true, y_pred), (yv_cont_true, yv_cont_pred), (ya_cont_true, ya_cont_pred)



def get_args():
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
    # parser.add_argument('--saved_log_path', type=str,
    #                     help='name of the saved model', required=True)

    # Get arguments from the parser in a dictionary,
    args = parser.parse_args()
    return args


def print_config(args):
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


def cont_result(y_true, y_pred):
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 计算CCC
    rho = np.corrcoef(y_true, y_pred)[0, 1]
    sigma_true = np.std(y_true)
    sigma_pred = np.std(y_pred)
    mu_true = np.mean(y_true)
    mu_pred = np.mean(y_pred)
    ccc = (2 * rho * sigma_true * sigma_pred) / (sigma_true ** 2 + sigma_pred ** 2 + (mu_true - mu_pred) ** 2)

    # 计算MAE
    mae = mean_absolute_error(y_true, y_pred)
    # 打印结果
    print("RMSE:", rmse)
    print("CCC:", ccc)
    print("MAE:", mae)


def main(args):
    output_file = os.path.join('./outputs', args.saved_model)
    print_config(args)

    if args.dataset == 'affectnet':
        resized_size = 227
    else:
        resized_size = 112

    if args.network == 'bregnext':
        resized_size = 112

    val_transforms = transforms.Compose([
        transforms.Resize((resized_size, resized_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    aligner = FaceAligner()

    '''加载测试数据集'''
    if args.dataset == 'affectnet':

        val_dataset = AffectNet_dataset(root_dir=args.val_image_dir, data_pkl=args.data, emb_pkl=args.emb,
                                        aligner=aligner, train=False, transform=val_transforms,
                                        crop_face=True)
    elif args.dataset == 'raf-db':

        val_dataset = RAF_DB_dataset(root_dir=args.val_image_dir, data_pkl=args.data, emb_pkl=args.emb, aligner=aligner,
                                     train=False, transform=val_transforms,
                                     crop_face=True)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.workers, pin_memory=True)

    model = MTC_GCN(adj_file=args.adj, input_size=resized_size)
    params = list(model.parameters())
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=0.001)

    # '''加载模型参数，模型：args.saved_model'''
    # checkpoint = torch.load(output_file)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



    model.cuda()
    print(model)





    # Define loss function
    if args.dataset == 'affectnet':
        weights = torch.FloatTensor(
            [3803 / 74874, 3803 / 134415, 3803 / 25459, 3803 / 14090, 3803 / 6378, 1, 3803 / 24882]).cuda()
        criterion_cat = torch.nn.CrossEntropyLoss(weight=weights)
    elif args.dataset == 'raf-db':
        # todo: raf_db的权重
        weights = torch.FloatTensor(
            [281 / 2524, 281 / 4772, 281 / 1982, 281 / 1290, 1, 281 / 717, 281 / 705]).cuda()
        criterion_cat = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion_cat = torch.nn.CrossEntropyLoss()

    criterion_cont = CCC_loss()

    val_losses = []
    val_cat_losses = []
    val_cont_losses = []

    val_loss, val_loss_cat, val_loss_cont, (y_val_true, y_val_pred), (yv_cont_true, yv_cont_pred), (ya_cont_true, ya_cont_pred) = \
        test_model_multi(args, val_dataloader, model, criterion_cat, criterion_cont, gcn=(args.model == 'emotion_gcn'))
    val_cat_losses.append(val_loss_cat)
    val_cont_losses.append(val_loss_cont)

    # Save losses to the corresponding lists
    val_losses.append(val_loss)

    # Convert preds and golds in a list.
    y_true = np.concatenate(y_val_true, axis=0)
    y_pred = np.concatenate(y_val_pred, axis=0)

    # Print metrics for current epoch
    print("My val loss is : {}".format(val_loss))
    print("My val categorical loss is : {}".format(val_loss_cat))
    print("My val continuous loss is : {}".format(val_loss_cont))

    cat_result(y_true, y_pred)

    yv_cont_true  = np.concatenate(yv_cont_true, axis=0)
    ya_cont_true = np.concatenate(ya_cont_true, axis=0)

    yv_cont_pred = np.concatenate(yv_cont_pred, axis=0)
    ya_cont_pred = np.concatenate(ya_cont_pred, axis=0)

    cont_result(yv_cont_true,yv_cont_pred)
    cont_result(ya_cont_true, ya_cont_pred)




if __name__ == '__main__':
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    main(args)
