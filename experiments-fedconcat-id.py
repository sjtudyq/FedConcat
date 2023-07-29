import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
from kmeanslimited import *
from sklearn.cluster import KMeans
import random

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=True, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for FedConcat')
    parser.add_argument('--encoder_round', type=int, default=50, help='Encoder communication round for FedConcat')
    parser.add_argument('--classifier_round', type=int, default=1000, help='Classifier communication round for FedConcat')
    parser.add_argument('--eps', type=float, default=0, help='Epsilon for differential privacy to protect label distribution')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            elif args.dataset == 'criteo':
                input_size = 12
                output_size = 2
                hidden_sizes = [16, 24, 32]    
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "simple-ae":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleAE_Cifar()
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleAE()
            elif args.dataset == 'celeba':
                net = SimpleAE()
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            net = ResNet50_cifar10()
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    for tmp in train_dataloader:
        for batch_idx, (x, target) in enumerate(tmp):
            sz = x.shape
            break
    
    logger.info(' ** Training complete **')
    return 0,0#train_acc, test_acc

def train_net_ae(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    #train_acc = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    #logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    #logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.MSELoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x).reshape(x.shape)
                loss = criterion(out, x)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    #train_acc = compute_accuracy(net, train_dataloader, device=device)
    #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    #logger.info('>> Training accuracy: %f' % train_acc)
    #logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    #return train_acc, test_acc
    return epoch_loss



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc

def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def train_net_encoder_classifier(net_id, net_encoder, net_classifier, train_dataloader, test_dataloader, epochs, lr, args_optimizer,edis=False, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        if args.dataset in ("cifar100", "tinyimagenet"):
            if args.partition == 'noniid-labeldir' and args.beta > 0.4:
                wd_factor = 0.005
            else:
                wd_factor = 0.001
            optimizer = optim.SGD([{'params': net_encoder.parameters()}, {'params': net_classifier.parameters()}], lr=lr, momentum=args.rho, weight_decay=wd_factor)
        else:
            optimizer = optim.SGD([{'params': net_encoder.parameters()}, {'params': net_classifier.parameters()}], lr=lr, momentum=args.rho, weight_decay=args.reg)
    #criterion = nn.MSELoss().to(device)
    classification_loss = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                encoded = net_encoder(x)
                #out = net_decoder(encoded)
                scores = net_classifier(encoded)

                #rec_loss = criterion(out, x.view(-1,3*32*32))
                loss = classification_loss(scores, target)

                #loss = rec_loss + cla_loss

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    logger.info(' ** Training complete **')

    if edis==False:
        return

    for tmp in train_dataloader:
        for batch_idx, (x, target) in enumerate(tmp):
            sz = x.shape
            break
    for i in range(100):
        if len(sz) == 4:
            noise = torch.rand((100,sz[1],sz[2],sz[3])).to(device)
        else:
            noise = torch.rand((100,sz[1],sz[2])).to(device)
        logits = net_classifier(net_encoder(noise)).cpu().detach()
        prob = torch.softmax(logits, dim=1)
        if i == 0:
            prob_add = torch.mean(prob, dim=0)
        else:
            prob_add += torch.mean(prob, dim=0)

    prob = prob_add/100
    prob = prob.tolist()
    
    logger.info(prob)

    return prob


def local_train_net_encoder_classifier(encoder_list, classifier_list, selected, args, net_dataidx_map, edis = False, test_dl = None, device="cpu"):
    avg_acc = 0.0
    estimated_dis = []

    for net_id in range(len(encoder_list)):
        if net_id not in selected:
            continue
        
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        encoder_list[net_id].to(device)
        classifier_list[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.dataset == "criteo":
            train_dl_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, need_test=False, net_id=net_id)
        else:
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        
        if edis:
            dis = train_net_encoder_classifier(net_id, encoder_list[net_id], classifier_list[net_id], train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,edis=True, device=device)
            estimated_dis.append(dis)
        else:
            train_net_encoder_classifier(net_id, encoder_list[net_id], classifier_list[net_id], train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        encoder_list[net_id].to("cpu")
        classifier_list[net_id].to("cpu")
    if edis:
        return estimated_dis

def train_net_classifier(net_id, net_encoder, net_classifier, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD([{'params': net_classifier.parameters()}], lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    net_classifier.to(device)
    net_encoder.to(device)

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net_classifier(net_encoder(x))
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

                if batch_idx == 2:
                    break  # only train 3 mini-batches

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    logger.info(' ** Training complete **')


def local_train_net_classifier(encoder, classifier_list, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id in range(len(classifier_list)):
        if net_id not in selected:
            continue
        
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        encoder.to(device)
        classifier_list[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        noise_level = args.noise / (args.n_parties - 1) * net_id
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs=dataidxs, noise_level=noise_level)
        n_epoch = 1

        train_net_classifier(net_id, encoder, classifier_list[net_id], train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para

def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= len(selected)
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc


    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map



if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    if args.dataset == 'criteo':
        args.n_parties = 10000

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train)) 

    client_distribution = np.zeros((args.n_parties, n_classes))
    for i in range(args.n_parties):
        for j in traindata_cls_counts[i].keys():
            client_distribution[i][j] = traindata_cls_counts[i][j] / len(net_dataidx_map[i])
            
    if args.eps > 0:
        for i in range(args.n_parties):
            lap = np.random.laplace(0,1/args.eps,n_classes)
            for j in range(n_classes):
                client_distribution[i][j] += lap[j]
    
        logger.info(client_distribution)

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)



    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))


            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)


    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))


            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)


        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to('cpu')
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            total_n = sum(n_list)
            #print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    #if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    #else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n


            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                #print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to('cpu')
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedae':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 10, args)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        cate_loss = []

        for party_id in range(args.n_parties):            
            dataidxs = net_dataidx_map[party_id]

            logger.info("Training network %s. n_training: %d" % (str(party_id), len(dataidxs)))
            noise_level = 0 #args.noise / (args.n_parties - 1) * party_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            
            # move the model to cuda device:
            nets[party_id] = torch.load("%d.pth" % party_id)
            net = nets[party_id]
            net.to(device)
            n_epoch = 50
            avg_loss = train_net_ae(party_id, net, train_dl_local, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)
            torch.save(net, "%d.pth" % party_id)
            cate_loss.append(1)
            #cate_loss.append(avg_loss)
            
        acc = compute_accuracy_ae(nets, test_dl_global, cate_loss)
        logger.info("test acc = %f" % acc)    

    elif args.alg == 'fedconcat':
        logger.info("Initializing nets")
        num_ftrs = 84
        if args.model == "vgg-9":
            num_ftrs = 512
        elif args.model == "resnet":
            num_ftrs = 2048
        encoder_list = []
        classifier_list = []
        for i in range(args.n_parties):
            if args.dataset in ("tinyimagenet", "cifar100", "cifar10", "svhn"):
                if args.model == "simple-cnn":
                    encoder = SimpleEncoder(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    encoder_list.append(encoder)
                elif args.model == "vgg-9":
                    encoder = ModerateEncoder()
                    encoder_list.append(encoder)
                elif args.model == "resnet":
                    encoder = ResNetEncoder()
                    encoder_list.append(encoder)
                if args.dataset == "cifar100":
                    classifier = SimpleClassifier(hidden_dim=num_ftrs, output_dim=100)
                elif args.dataset == "tinyimagenet":
                    classifier = SimpleClassifier(hidden_dim=num_ftrs, output_dim=200)
                else:
                    classifier = SimpleClassifier(hidden_dim=num_ftrs, output_dim=10)
                classifier_list.append(classifier)
            else:
                encoder = SimpleEncoder(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                encoder_list.append(encoder)
                classifier = SimpleClassifier(hidden_dim=num_ftrs, output_dim=10)
                classifier_list.append(classifier)

        num_K = args.n_clusters
        encoder_global_para = [encoder_list[0].state_dict() for i in range(num_K)]
        classifier_global_para = [classifier_list[0].state_dict() for i in range(num_K)]

        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        
        logger.info('comm round 0')
        selected = [i for i in range(args.n_parties)]
        for idx in selected:
            encoder_list[idx].load_state_dict(encoder_global_para[0])
            classifier_list[idx].load_state_dict(classifier_global_para[0])
        estimated_dis = local_train_net_encoder_classifier(encoder_list, classifier_list, selected, args, net_dataidx_map, edis=True, test_dl = test_dl_global, device=device)

        logger.info(estimated_dis)

        estimator = KMeans(n_clusters=num_K)
        estimator.fit(estimated_dis)
        assign = estimator.labels_
        if args.dataset in ("cifar100", "tinyimagenet"):
            assign = constrained_kmeans(np.array(estimated_dis),num_K,int(1.2*args.n_parties/args.n_clusters))
        logger.info(assign)
        group = [[] for i in range(num_K)]
        for i in range(args.n_parties):
            group[assign[i]].append(i)

        logger.info(group)

        for i in range(num_K):
            selected = group[i]
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]


            for idx in range(len(selected)):
                net_para = encoder_list[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        encoder_global_para[i][key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        encoder_global_para[i][key] += net_para[key] * fed_avg_freqs[idx]

            for idx in range(len(selected)):
                net_para = classifier_list[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        classifier_global_para[i][key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        classifier_global_para[i][key] += net_para[key] * fed_avg_freqs[idx]
        
        '''
        # code for combining the shallow layers
        selected = [i for i in range(args.n_parties)]
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [sum([len(net_dataidx_map[r]) for r in list(set(selected).intersection(set(group[c])))]) / total_data_points for c in range(num_K)]

        net_para = encoder_list[0].cpu().state_dict()
        for key in net_para:
            if "fc2" not in key:
                encoder_global_para[0][key] *= fed_avg_freqs[0]
        for c in range(1,num_K):
            for key in net_para:
                if "fc2" not in key:
                    encoder_global_para[0][key] += encoder_global_para[c][key] * fed_avg_freqs[c]

        for c in range(1,num_K):
            for key in net_para:
                if "fc2" not in key:
                    encoder_global_para[i][key] = encoder_global_para[0][key]
        '''
        
        for round in range(1,args.encoder_round):
            logger.info("in comm round:" + str(round))
            top = int(args.n_parties * args.sample)
            participation = np.random.permutation(np.arange(args.n_parties))[:top]

            for i in range(num_K):
                selected = []
                for x in group[i]:
                    if x in participation:
                        selected.append(x)
                if len(selected) == 0:
                    continue

                for idx in selected:
                    encoder_list[idx].load_state_dict(encoder_global_para[i])
                    classifier_list[idx].load_state_dict(classifier_global_para[i])

                local_train_net_encoder_classifier(encoder_list, classifier_list, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)

                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]


                for idx in range(len(selected)):
                    net_para = encoder_list[selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            encoder_global_para[i][key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            encoder_global_para[i][key] += net_para[key] * fed_avg_freqs[idx]

                for idx in range(len(selected)):
                    net_para = classifier_list[selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            classifier_global_para[i][key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            classifier_global_para[i][key] += net_para[key] * fed_avg_freqs[idx]            
            
            '''
            # code for combining the shallow layers
            selected = participation
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [sum([len(net_dataidx_map[r]) for r in list(set(selected).intersection(set(group[c])))]) / total_data_points for c in range(num_K)]
            #logger.info(fed_avg_freqs)

            net_para = encoder_list[0].cpu().state_dict()
            for key in net_para:
                #logger.info(key)
                if "fc2" not in key:
                    encoder_global_para[0][key] *= fed_avg_freqs[0]
            for c in range(1,num_K):
                for key in net_para:
                    if "fc2" not in key:
                        encoder_global_para[0][key] += encoder_global_para[c][key] * fed_avg_freqs[c]

            for c in range(1,num_K):
                for key in net_para:
                    if "fc2" not in key:
                        encoder_global_para[i][key] = encoder_global_para[0][key]
            '''
                        
        encoder_selected = []
        for i in range(num_K):
            encoder_selected.append(encoder_list[i])
            encoder_selected[i].load_state_dict(encoder_global_para[i])
            encoder_selected[i].to(device)
        classifier_selected = []
        for i in range(num_K):
            classifier_selected.append(classifier_list[i])
            classifier_selected[i].load_state_dict(classifier_global_para[i])
            classifier_selected[i].to(device)

        encoder_all = CombineAllModel(encoder_selected)
        if args.dataset == "cifar100":
            output_dim = 100
        elif args.dataset == "tinyimagenet":
            output_dim = 200
        else:
            output_dim = 10
        larger_classifier_global = SimpleClassifier(hidden_dim=num_ftrs*num_K, output_dim=output_dim)
        
        start_from_classifier_weight = False # if set to True, the final classifier layer will be initialized as the previous cluster model's classifier layers (to speed up convergence).
        if args.dataset in ("cifar100", "tinyimagenet"):
            start_from_classifier_weight = True
        if start_from_classifier_weight:
            # Retrieve the weights from each model in classifier_global_para
            weights = [model.fc.weight for model in classifier_selected]
            biases = [model.fc.bias for model in classifier_selected]

            # Concatenate along the dimension of number of neurons in the hidden layer
            # The weights are transposed before concatenating because in PyTorch, the Linear layer's weight shape is (output_dim, input_dim)
            weights_tensor = torch.cat([w.t() for w in weights], dim=0).t()
            biases_tensor = sum(biases)
            logger.info(weights_tensor.shape)
            logger.info(biases_tensor.shape)

            # Create a state dictionary to load into the larger_classifier_global
            state_dict = {
                'fc.weight': weights_tensor,
                'fc.bias': biases_tensor,
            }

            # Load the state dict into the model
            larger_classifier_global.load_state_dict(state_dict)
            logger.info("loading initial classifier parameters done")
            
        larger_classifier_global_para = larger_classifier_global.cpu().state_dict()

        larger_classifier_list = []
        for t in range(args.n_parties):
            larger_classifier_list.append(SimpleClassifier(hidden_dim=num_ftrs*num_K, output_dim=output_dim))

        for round in range(args.classifier_round):
            
            logger.info("in comm round:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            for idx in selected:
                larger_classifier_list[idx].load_state_dict(larger_classifier_global_para)

            local_train_net_classifier(encoder_all, larger_classifier_list, selected, args, net_dataidx_map, device=device)

            #total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            #fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
            fed_avg_freqs = [1/len(selected) for r in selected]  # This is because we move 3 steps for all clients, so this average should be unweighted.


            for idx in range(len(selected)):
                net_para = larger_classifier_list[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        larger_classifier_global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        larger_classifier_global_para[key] += net_para[key] * fed_avg_freqs[idx]

            larger_classifier_global.load_state_dict(larger_classifier_global_para)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            
        
            global_model = CombineModel(encoder_all, larger_classifier_global)
            global_model.to(device)

            #train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc = compute_accuracy(global_model, test_dl_global, device=device)


            #logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)     


    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs

        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)

   
