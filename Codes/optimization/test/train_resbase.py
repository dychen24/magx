from tensorboard.backend.event_processing import event_accumulator
import os
import json
import time
import numpy as np
import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

from src.dl import load_model, save_model, data_set, FC_Net, ResNet18, ResNet152, ResBase18


# %%
time_start = time.time()
config = json.load(open("config.json"))
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

LR = config['lr']
EPOCH = config['epoch']
WD = config['Weight_Decay']
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
parser.add_argument("--epoch", default=0, type=int,
                    help="The epoch to be tested")
parser.add_argument("--optimizer", default='sgd', type=str,
                    help="The used optimizer")
parser.add_argument("--lr", default=LR, type=float,
                    help="The epoch to be tested")
parser.add_argument("--name", default='Resnet_FE_{}'.format(LR), type=str,
                    help="Whether to test after training")
args = parser.parse_args()
LR = args.lr
DEVICE = t.device(config["DEVICE"]+':'+args.gpu)
DataSet = data_set()
project_name = 'ResBase_FE_{}_{}'.format(args.optimizer, LR)
# using K-fold
n_splits = 5
np.random.seed(1998)
kf = KFold(n_splits=n_splits)
idx = np.arange(len(DataSet))
np.random.shuffle(idx)
print(project_name, kf.get_n_splits(idx))
# shuffle the data before
for K_idx, [train_idx, test_idx] in enumerate(kf.split(idx)):
    writer = SummaryWriter(
        'result/logs/{}_{}_Fold'.format(project_name, K_idx+1))

    train_data, test_data = data_set(train_idx), data_set(test_idx)
    # train_data.data_argumentation()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    model = ResBase18(2).to(DEVICE)

    if args.optimizer == 'sgd':
        print('use sgd')
        optimizer = t.optim.SGD(model.parameters(), lr=LR)
    elif args.optimizer == 'momentum':
        print('use momentum ')
        optimizer = t.optim.SGD(model.parameters(), lr=LR,
                                momentum=0.9, weight_decay=0.01)
    elif args.optimizer == 'adam':
        print('use adam')
        optimizer = t.optim.Adam(model.parameters(), lr=LR)
    print(optimizer.param_groups[0]['lr'])
    # optimizer = t.optim.Adam(model.parameters())
    criterian = t.nn.CrossEntropyLoss().to(DEVICE)

    # Test the train_loader
    for epoch in tqdm(range(EPOCH)):
        model = model.train()
        train_loss = 0
        correct = 0
        for batch_idx, [data, label] in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            data = data.unsqueeze(1)
            out = model(data).squeeze()
            loss = criterian(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(label.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / len(train_loader.dataset)

        # train_l.append(train_loss)
        # train_a.append(train_acc)

        # print('\nEpoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     epoch, train_loss, correct, len(train_loader.dataset), train_acc))

        model = model.eval()

        with t.no_grad():
            # Test the test_loader
            test_loss = 0
            correct = 0
            for batch_idx, [data, label] in enumerate(test_loader):
                data, label = data.to(DEVICE), label.to(DEVICE)
                data = data.unsqueeze(1)
                out = model(data)
                # monitor the upper and lower boundary of output
                # out_max = t.max(out)
                # out_min = t.min(out)
                # out = (out - out_min) / (out_max - out_min)
                test_loss += criterian(out, label)
                pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct += pred.eq(label.view_as(pred)).sum().item()
            # store params
            for name, param in model.named_parameters():
                writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch)

            test_loss /= len(test_loader.dataset)
            test_acc = 100. * correct / len(test_loader.dataset)

            # test_l.append(test_loss)
            # test_a.append(test_acc)

            # print('Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            # epoch, test_loss, correct, len(test_loader.dataset), test_acc))
        if (epoch+1) % 50 == 0:
            save_model(model, epoch, '{}_{}_folds'.format(
                project_name, K_idx+1))
        # eval_model_new_thread(epoch, 0)
        # LZX pls using the following code instead
        # multiprocessing.Process(target=eval_model(epoch, '0'), args=(multiprocess_idx,))
        # multiprocess_idx += 1
        writer.add_scalar('Training/Training_Loss', train_loss, epoch)
        writer.add_scalar('Training/Training_Acc', train_acc, epoch)
        writer.add_scalar('Testing/Testing_Loss', test_loss, epoch)
        writer.add_scalar('Testing/Testing_Acc', test_acc, epoch)
    writer.close()

training_loss = np.zeros(EPOCH, dtype=np.float)
testing_loss = np.zeros(EPOCH, dtype=np.float)
training_acc = np.zeros(EPOCH, dtype=np.float)
testing_acc = np.zeros(EPOCH, dtype=np.float)

# compute the mean acc and loss
dirs = ['result/logs/{}_{}_Fold'.format(project_name, i+1) for i in range(5)]
writer = SummaryWriter('result/logs/{}'.format(project_name))
for dir in dirs:
    try:
        data = os.listdir(dir)
    except:
        break
    print(data)
    ea = event_accumulator.EventAccumulator(os.path.join(dir, data[0]))
    ea.Reload()
    # print(ea.scalars.Keys())
    train_loss = ea.scalars.Items('Training/Training_Loss')
    training_loss += np.array([i.value for i in train_loss])

    train_acc = ea.scalars.Items('Training/Training_Acc')
    training_acc += np.array([i.value for i in train_acc])

    test_loss = ea.scalars.Items('Testing/Testing_Loss')
    testing_loss += np.array([i.value for i in test_loss])

    test_acc = ea.scalars.Items('Testing/Testing_Acc')
    testing_acc += np.array([i.value for i in test_acc])

training_loss /= n_splits
training_acc /= n_splits
testing_loss /= n_splits
testing_acc /= n_splits
print(training_acc)
for epoch in range(EPOCH):
    writer.add_scalar('Training/Training_Loss', training_loss[epoch], epoch)
    writer.add_scalar('Training/Training_Acc', training_acc[epoch], epoch)
    writer.add_scalar('Testing/Testing_Loss', testing_loss[epoch], epoch)
    writer.add_scalar('Testing/Testing_Acc', testing_acc[epoch], epoch)
writer.close()


# Final Train
writer = SummaryWriter(
    'result/logs/{}_Final'.format(project_name))
train_data = data_set(idx)
train_loader = DataLoader.DataLoader(
    train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
model = ResBase18(2).to(DEVICE)

if args.optimizer == 'sgd':
    print('use sgd')
    optimizer = t.optim.SGD(model.parameters(), lr=LR)
elif args.optimizer == 'momentum':
    print('use momentum ')
    optimizer = t.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
elif args.optimizer == 'adam':
    print('use adam')
    optimizer = t.optim.Adam(model.parameters(), lr=LR)
print(optimizer.param_groups[0]['lr'])
# optimizer = t.optim.Adam(model.parameters())
criterian = t.nn.CrossEntropyLoss().to(DEVICE)

# Test the train_loader
for epoch in tqdm(range(EPOCH)):
    model = model.train()
    train_loss = 0
    correct = 0
    for batch_idx, [data, label] in enumerate(train_loader):
        data, label = data.to(DEVICE), label.to(DEVICE)
        data = data.unsqueeze(1)
        out = model(data).squeeze()
        loss = criterian(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(label.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    if (epoch+1) % 2 == 0:
        save_model(model, epoch, '{}_{}_final'.format(
            project_name, K_idx+1))
        # eval_model_new_thread(epoch, 0)
        # LZX pls using the following code instead
        # multiprocessing.Process(target=eval_model(epoch, '0'), args=(multiprocess_idx,))
        # multiprocess_idx += 1
    writer.add_scalar('Training/Training_Loss', train_loss, epoch)
    writer.add_scalar('Training/Training_Acc', train_acc, epoch)
writer.close()
