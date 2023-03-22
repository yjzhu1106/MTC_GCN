from datetime import datetime

import torch
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

def train_model_single(dataloader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):
        inputs, labels, _, _ = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        y_pred.append(preds.cpu().numpy())
        y_true.append(labels.cpu().numpy())

    return running_loss / index, (y_true, y_pred)


def eval_model_single(dataloader, model, criterion):
    model.eval()
    running_loss = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            inputs, labels, _, _ = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())

            running_loss += loss.data.item()

    return running_loss / index, (y_true, y_pred)
    
def train_model_multi(args, dataloader, model, criterion_cat, criterion_cont, optimizer, gcn=False,epoch=1):
    model.train()
    running_loss = 0.0
    running_loss_cat = 0.0
    running_loss_cont = 0.0
    y_pred = []
    y_true = []

    i = 0
    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):
        if batch is None:
            print("Error: batch is None")
            continue
        inputs, labels, labels_cont, inp = batch

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_cont = labels_cont.to(device)
        inp = inp.to(device)

        optimizer.zero_grad()

        if gcn:
            outputs_cat, outputs_cont = model(inputs, inp)
        else:
            outputs_cat, outputs_cont = model(inputs)

        loss_cat = criterion_cat(outputs_cat, labels)
        loss_cont = criterion_cont(outputs_cont.double(), labels_cont.double())
        _, preds = torch.max(outputs_cat, 1)
        loss = loss_cat + loss_cont

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_loss_cat += loss_cat.data.item()
        running_loss_cont += loss_cont.data.item()

        y_pred.append(preds.cpu().numpy())
        y_true.append(labels.cpu().numpy())
        y_train_true = np.concatenate(y_true, axis=0)
        y_train_pred = np.concatenate(y_pred, axis=0)

        list = ["%s" % datetime.now(),
                '%s'% epoch,
                '%s' % index,
                '%.4f' % accuracy_score(y_train_true, y_train_pred),
                '%.4f' % (running_loss/ index),
                '%.4f' % (running_loss_cat/ index),
                '%.4f' % (running_loss_cont/ index),
                '',
                '',
                '',
                '',
                ]

        data = pd.DataFrame([list])


        data.to_csv(args.saved_log_path + '/train_log.csv', mode='a', header=False, index=False)  # 路径可以根据需要更改
        print(f"Epoch {epoch + 1}, batch: {index}/{len(dataloader)}: acc={accuracy_score(y_train_true, y_train_pred):.4f}, loss={(running_loss/ index):.4f}, loss_cat={(running_loss_cat/ index):.4f}, "
              f"loss_cont={(running_loss_cont/ index):.4f}")

    return running_loss / index, running_loss_cat / index, running_loss_cont / index, (y_true, y_pred)


def eval_model_multi(args, dataloader, model, criterion_cat, criterion_cont, gcn=False):
    model.eval()
    running_loss = 0.0
    running_loss_cat = 0.0
    running_loss_cont = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            inputs, labels, labels_cont, inp = batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_cont = labels_cont.to(device)
            inp = inp.to(device)

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

            running_loss += loss.data.item()
            running_loss_cat += loss_cat.data.item()
            running_loss_cont += loss_cont.data.item()
            y_train_true = np.concatenate(y_true, axis=0)
            y_train_pred = np.concatenate(y_pred, axis=0)

            list = ["%s" % datetime.now(),
                    '' ,
                    '%s' % index,
                    '',
                    '',
                    '',
                    '',
                    '%.4f' % accuracy_score(y_train_true, y_train_pred),
                    '%.4f' % (running_loss / index),
                    '%.4f' % (running_loss_cat / index),
                    '%.4f' % (running_loss_cont / index),
                    ]

            data = pd.DataFrame([list])
            data.to_csv(args.saved_log_path + '/train_log.csv', mode='a', header=False,
                        index=False)  # 路径可以根据需要更改
    return running_loss / index, running_loss_cat / index, running_loss_cont / index, (y_true, y_pred)
