import tqdm
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
from model import *
from dataloader import get_dataloader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os
import tensorboardX
import yaml
from os.path import join as pjoin
import argparse
import torch.nn.init as init
import math
from torch.optim import lr_scheduler

def train(model, criterion, optimizer, train_loader, val_loader, epochs, writer, dir):
    
    best_test_loss = np.inf
    iteration = 0
    eval_frequency = 400

    with tqdm.trange(0, epochs, desc="epochs") as tbar, tqdm.tqdm(
            total=eval_frequency, leave=False, desc="train"
        ) as pbar:
        for epoch in tbar:
            model.train()
            train_loss = []
            for inputs, targets in train_loader:
                
                iteration += 1
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                pbar.update()
                pbar.set_postfix(dict(total_it=iteration))
                tbar.refresh()

                if iteration % eval_frequency == 0:
                    pbar.close()

                    # Get train loss and test loss
                    writer.add_scalar('train/loss', np.mean(train_loss), iteration)
                    train_loss = []
                    
                    precision_lst, recall_lst, F_beta_lst, test_loss = test(model, val_loader, criterion)

                    for i in range(3):
                        writer.add_scalar('test/precision_%d' % i, precision_lst[i], iteration)
                        writer.add_scalar('test/recall_%d' % i, recall_lst[i], iteration)
                        writer.add_scalar('test/F_beta_%d' % i, F_beta_lst[i], iteration)

                    writer.add_scalar('test/loss', test_loss, iteration)

                    if test_loss < best_test_loss:
                        torch.save(model, dir + 'best_val_model')
                        best_test_loss = test_loss
                        print('model saved')
                    pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc="train"
                        )
                    pbar.set_postfix(dict(total_it=iteration))
                model.train()

                
    torch.save(model, f'final_model')
    return

def evaluate(all_targets, all_predictions):
    print('Overall accuracy: %.04f' % accuracy_score(all_targets, all_predictions))
    reports = classification_report(all_targets, all_predictions, digits=4, output_dict=True)
    precision_lst = []
    recall_lst = []
    F_beta_lst = []
    for i in range(3):
        precision = reports['%d' % i]['precision']
        recall = reports['%d' % i]['recall']
        precision_lst.append(precision)
        recall_lst.append(recall)

        if precision + recall == 0:
            print('label %d--------precision: %.04f recall: %.04f f0.5: NaN' % (i, precision, recall))
            F_beta_lst.append(0)
        else:
            F_beta = (1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall)
            print('label %d--------precision: %.04f recall: %.04f f0.5: %.04f' % (i, precision, recall, F_beta))
            F_beta_lst.append(F_beta)
    return precision_lst, recall_lst, F_beta_lst

def test(model, dataloader, criterion):
    model.eval()

    all_targets = []
    all_predictions = []
    test_loss = []

    for inputs, targets in dataloader:

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss.append(loss.item())
        
        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)    
    all_predictions = np.concatenate(all_predictions)
    precision_lst, recall_lst, F_beta_lst = evaluate(all_targets, all_predictions)
    test_loss = np.mean(test_loss)
    return precision_lst, recall_lst, F_beta_lst, test_loss

def get_model(name):
    if name == 'DeepLOB':
        model = DeepLOB()
    elif name == 'MLP':
        model = MLP()
    elif name == 'Transformer':
        model = Transformer()
    elif name == 'deeplob':
        model = deeplob()
    elif name == 'deeplob_bn':
        model = deeplob_bn()
    else:
        raise NotImplementedError
    return model

def weights_init(m):                                              
    classname = m.__class__.__name__                              
    if classname.find('Conv') != -1:                              
        init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
    elif classname.find('BatchNorm') != -1:                       
        init.normal_(m.weight.data, 1.0, 0.02)                 
        init.constant_(m.bias.data, 0)
    return 

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    f = open(pjoin('./', 'config', args.config), 'r')
    cfg = yaml.load(f, Loader=yaml.FullLoader)

    dir = cfg['dir_pth']

    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = tensorboardX.SummaryWriter(log_dir=dir)
    
    model = get_model(cfg['model'])
    model.apply(weights_init)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight = torch.Tensor([cfg['loss_weight'],1,cfg['loss_weight']]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)#, momentum=0.9, weight_decay = 1e-5)

    train_loader, val_loader, test_loader = get_dataloader(cfg['period'], device, batch_size=512, days=cfg['days'])

    train(model, criterion, optimizer, train_loader, val_loader, epochs=100, writer=writer, dir=dir)
    test(model, test_loader, criterion)