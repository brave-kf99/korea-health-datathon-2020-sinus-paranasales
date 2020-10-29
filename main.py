import os, sys
import argparse
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms

import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from efficientnet_pytorch import EfficientNet
from copy import deepcopy
# from apex import amp

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

# custom files
import focal_loss
import pns_dataset
import ensemble

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def bind_model(model, device):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(data):  ## test mode
        # custom dataset
        transform_test = A.Compose([
                                A.Resize(image_size*2, image_size),
                                A.CenterCrop(image_size, image_size),
                                A.Normalize()
                            ])
        
        dataset_pns_test = pns_dataset.PNSTestDataset(data, transform=transform_test)

        result_infer = []

        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        with torch.no_grad():
            for idx in range(len(dataset_pns_test)):
                X = dataset_pns_test[idx]
                X = X.to(device)
                pred = model.forward(X)
                prob, pred_cls = torch.max(pred, 1)
                result_infer.append(pred_cls.item())

        print('Prediction done!\n Saving the result...')
        print('First 50 of pred: {}'.format(result_infer[:50]))
        return result_infer

    nsml.bind(save=save, load=load, infer=infer)



def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, w_:2*w_]
        r_img = img_whole[:, :w_]

        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        
        if l_cls=='0':
            img.append(l_img);      lb.append(int(l_cls))

        if r_cls=='0':
            img.append(r_img);      lb.append(int(r_cls))

        if l_cls=='1':
            for i in range(7):
                img.append(l_img);      lb.append(int(l_cls))
                
        if r_cls=='1':
            for i in range(7):
                # r_img = horizontal_flip(r_img,True) # horizontal flip to left side face
                img.append(r_img);      lb.append(int(r_cls))
        
        if l_cls=='2' :
            for i in range(17):
                img.append(l_img);      lb.append(int(l_cls))
        
        if r_cls=='2' :
            for i in range(17):
                # r_img = horizontal_flip(r_img,True) # horizontal flip to left side face
                img.append(r_img);      lb.append(int(r_cls))

        if l_cls=='3':
            for i in range(28):
                img.append(l_img);      lb.append(int(l_cls))
        if r_cls=='3':
            for i in range(28):
                # r_img = horizontal_flip(r_img,True) # horizontal flip to left side face
                img.append(r_img);      lb.append(int(r_cls))

    print(len(img), 'data with label 0-3 loaded!')
    return img, lb
def ClassDataLoad(impath, data_class):
    img = []
    lb = []
    multiply = 1
    if data_class == 1:
        multiply = 7
    elif data_class == 2:
        multiply = 17
    elif data_class == 3:
        multiply = 28


    data_class = str(int(data_class))
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, w_:2*w_] ###########################################changed
        r_img = img_whole[:, :w_]     ###########################################changed

        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls==data_class:
            for i in range(multiply):
                img.append(l_img);      lb.append(int(l_cls))
        if r_cls==data_class:
            for i in range(multiply):
                r_img = horizontal_flip(r_img,True) # horizontal flip to left side face
                img.append(r_img);      lb.append(int(r_cls))
    print(len(img), 'data with label 0-3 loaded!')
    return img, lb

def ImagePreprocessing(img):
    # 자유롭게 작성
    # h, w = IMSIZE
    h, w = 512, 512
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        tmp = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA)
        tmp = tmp / 255.
        img[i] = tmp
    print(len(img), 'images processed!')
    return img


def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=300)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--image_size', type=int, default=300)      # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개
    args.add_argument('--learning_rate', type=float, default=0.0001)  # learning rate 설정
    args.add_argument('--model_type', type=str, default='efficientnet-b3') # set model type
    args.add_argument('--use_pretrained', type=bool, default=True)     # set use pretrained or not

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.image_size, config.num_classes, config.learning_rate, config.model_type, config.use_pretrained, config.pause, config.mode


cosine_epo = 10
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    print(GPU_NUM)
    nb_epoch, batch_size, image_size, num_classes, learning_rate, model_type, use_pretrained, ifpause, ifmode = ParserArguments(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #####   Model   #####

    # Set seperate models

    modelEb4 = EfficientNet.from_pretrained('efficientnet-b4',in_channels=3,num_classes=4)
    modelEb3 = EfficientNet.from_pretrained('efficientnet-b3',in_channels=3,num_classes=4)
    modelRes18 = models.resnet18(pretrained=True)
    num_ftrs = modelRes18.fc.in_features
    modelRes18.fc = nn.Linear(num_ftrs, 4) # the last fc layer

    modelEb4.to(device)
    modelEb3.to(device)
    modelRes18.to(device) 

    model = ensemble.Ensemble(modelEb4, modelEb3, modelRes18).to(device)
    bind_model(model, device)

    criterion = focal_loss.FocalLoss(device).to(device)

    optimizerEb4 = torch.optim.Adam(modelEb4.parameters(), lr=learning_rate)
    optimizerEb3 = torch.optim.Adam(modelEb3.parameters(), lr=learning_rate)
    optimizerRes18 = torch.optim.Adam(modelRes18.parameters(), lr=learning_rate)
    scheduler_cosineEb4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerEb4, cosine_epo)
    scheduler_cosineEb3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerEb3, cosine_epo)
    scheduler_cosineRes18 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerRes18, cosine_epo)

    if ifpause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())

    if ifmode == 'train':  ## for train mode

        print('Training start ...')
        # 자유롭게 작성

        transform_train = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightness(limit=0.2, p=0.75),
                            A.RandomContrast(limit=0.2, p=0.75),
                            A.OneOf([
                                A.MotionBlur(blur_limit=5),
                                A.MedianBlur(blur_limit=5),
                                A.GaussianBlur(blur_limit=5),
                                A.GaussNoise(var_limit=(5.0, 30.0)),
                            ], p=0.7),
                            A.Resize(image_size*2, image_size),
                            A.CenterCrop(image_size, image_size),
                            A.Normalize()
                        ])

        transform_val = A.Compose([
                                A.Resize(image_size*2, image_size),
                                A.CenterCrop(image_size, image_size),
                                A.Normalize()
                            ])

        imdir = os.path.join(DATASET_PATH, 'train')
        img_pns_all, label_pns_all = DataLoad(imdir)
        img_pns_train, img_pns_val, labels_train, labels_val = train_test_split(img_pns_all, label_pns_all, test_size=0.20, random_state=42)

        dataset_pns_train = pns_dataset.PNSDataset(img_pns_train, labels_train, transform=transform_train)
        dataset_pns_val = pns_dataset.PNSDataset(img_pns_val, labels_val, transform=transform_val)

        batch_train = DataLoader(dataset_pns_train, batch_size=batch_size, shuffle=True, drop_last=True)
        batch_val = DataLoader(dataset_pns_val, batch_size=1, shuffle=False, drop_last=False)


        #####   Training loop   #####
        STEP_SIZE_TRAIN = len(dataset_pns_train) // batch_size
        print('\n\n STEP_SIZE_TRAIN= {}\n\n'.format(STEP_SIZE_TRAIN))
        t0 = time.time()
        for epoch in range(nb_epoch):

            pred_array_tr = []
            label_array_tr = []

            class_correct = [0, 0, 0, 0]
            class_total = [0, 0, 0, 0]
            class_correctEb4 = [0, 0, 0, 0]
            class_totalEb4 = [0, 0, 0, 0]
            class_correctEb3 = [0, 0, 0, 0]
            class_totalEb3 = [0, 0, 0, 0]
            class_correctRes18 = [0, 0, 0, 0]
            class_totalRes18 = [0, 0, 0, 0]

            t1 = time.time()
            print('Model fitting ...')
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))
            a, a_val, tp, tp_val = 0, 0, 0, 0
            for i, (x_tr, y_tr) in enumerate(batch_train):
                x_tr0, y_tr0 = deepcopy(x_tr), deepcopy(y_tr)
                x_tr1, y_tr1 = deepcopy(x_tr), deepcopy(y_tr)
                x_tr2, y_tr2 = deepcopy(x_tr), deepcopy(y_tr)
                x_tr, y_tr = x_tr.to(device), y_tr.to(device)
                x_tr0, y_tr0 = x_tr0.to(device), y_tr0.to(device)
                x_tr1, y_tr1 = x_tr1.to(device), y_tr1.to(device)
                x_tr2, y_tr2 = x_tr2.to(device), y_tr2.to(device)


                # Eb4
                optimizerEb4.zero_grad()
                predEb4 = modelEb4(x_tr0)
                lossEb4 = criterion(predEb4, y_tr0)
                lossEb4.backward()
                optimizerEb4.step()

                probEb4, pred_clsEb4 = torch.max(predEb4, 1)
                resultEb4 = (pred_clsEb4 == y_tr0).squeeze()

                # Eb3
                optimizerEb3.zero_grad()
                predEb3 = modelEb3(x_tr1)
                lossEb3 = criterion(predEb3, y_tr1)

                lossEb3.backward()
                optimizerEb3.step()

                probEb3, pred_clsEb3 = torch.max(predEb3, 1)
                resultEb3 = (pred_clsEb3 == y_tr1).squeeze()

                # Res18
                optimizerRes18.zero_grad()
                predRes18 = modelRes18(x_tr2)
                lossRes18 = criterion(predRes18, y_tr2)

                lossRes18.backward()
                optimizerRes18.step()

                probRes18, pred_clsRes18 = torch.max(predRes18, 1)
                resultR2es18 = (pred_clsRes18 == y_tr2).squeeze()


                # ensemble model
                model = ensemble.Ensemble(modelEb4, modelEb3, modelRes18).to(device)
                bind_model(model, device)
                pred = model(x_tr)
                loss = criterion(pred, y_tr)

                prob, pred_cls = torch.max(pred, 1)
                a += y_tr.size(0)
                tp += (pred_cls == y_tr).sum().item()

                result = (pred_cls == y_tr).squeeze()

                pred_array_tr.append(pred_cls.data.cpu().numpy())
                label_array_tr.append(y_tr.data.cpu().numpy())
                # get result per class
                for c in range(batch_size):
                    c_label = y_tr[c]
                    class_correct[int(c_label)] += result[c].item()
                    class_total[int(c_label)] += 1
                    class_correctEb4[int(c_label)] += result[c].item()
                    class_totalEb4[int(c_label)] += 1
                    class_correctEb3[int(c_label)] += result[c].item()
                    class_totalEb3[int(c_label)] += 1
                    class_correctRes18[int(c_label)] += result[c].item()
                    class_totalRes18[int(c_label)] += 1
                
            print("[*] train shape/min/max {}/{}/{} ".format(x_tr.shape, torch.min(x_tr), torch.max(x_tr)))
            scheduler_cosineEb4.step()
            scheduler_cosineEb3.step()
            scheduler_cosineRes18.step()


            f1s_tr = []
            pred_array_tr = np.array(pred_array_tr)
            label_array_tr = np.array(label_array_tr)
            for result_class in range(4):
                pred_array_tr_tmp = deepcopy(pred_array_tr)
                label_array_tr_tmp = deepcopy(label_array_tr)

                pred_array_tr_tmp[pred_array_tr_tmp == result_class] = 100
                pred_array_tr_tmp[pred_array_tr_tmp != 100] = 0
                pred_array_tr_tmp[pred_array_tr_tmp == 100] = 1

                label_array_tr_tmp[label_array_tr_tmp == result_class] = 100
                label_array_tr_tmp[label_array_tr_tmp != 100] = 0
                label_array_tr_tmp[label_array_tr_tmp == 100] = 1

                f1 = f1_score(pred_array_tr_tmp.flatten(), label_array_tr_tmp.flatten(), pos_label=1)
                f1s_tr.append(f1)
            f1_weighted_tr = ( (1 * np.sum(f1s_tr[0])) + (2 * np.sum(f1s_tr[1])) + (3 * np.sum(f1s_tr[2])) + (4 * np.sum(f1s_tr[3])) ) / 10

            class_correct_val = [0, 0, 0, 0]
            class_total_val = [0, 0, 0, 0]
            with torch.no_grad():
                pred_array_val = []
                label_array_val = []
                for j, (x_val, y_val) in enumerate(batch_val):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    pred_val = model(x_val)
                    loss_val = criterion(pred_val, y_val)
                    prob_val, pred_cls_val = torch.max(pred_val, 1)
                    a_val += y_val.size(0)
                    tp_val += (pred_cls_val == y_val).sum().item()
                    result_val = (pred_cls_val == y_val).squeeze()
                    c_label_val = y_val.item()
                    class_correct_val[int(c_label_val)] += result_val.item()
                    class_total_val[int(c_label_val)] += 1
                    pred_array_val.append(pred_cls_val.data.cpu().numpy())
                    label_array_val.append(y_val.data.cpu().numpy())

                    # if j == 2:
                    #     break

                f1s_val = []
                pred_array_val = np.array(pred_array_val)
                label_array_val = np.array(label_array_val)

                for result_class in range(4):
                    pred_array_val_tmp = deepcopy(pred_array_val)
                    label_array_val_tmp = deepcopy(label_array_val)

                    pred_array_val_tmp[pred_array_val_tmp == result_class] = 100
                    pred_array_val_tmp[pred_array_val_tmp != 100] = 0
                    pred_array_val_tmp[pred_array_val_tmp == 100] = 1

                    label_array_val_tmp[label_array_val_tmp == result_class] = 100
                    label_array_val_tmp[label_array_val_tmp != 100] = 0
                    label_array_val_tmp[label_array_val_tmp == 100] = 1
                    f1 = f1_score(pred_array_val_tmp.flatten(), label_array_val_tmp.flatten(), pos_label=1)
                    f1s_val.append(f1)
                f1_weighted_val = ( (1 * np.sum(f1s_val[0])) + (2 * np.sum(f1s_val[1])) + (3 * np.sum(f1s_val[2])) + (4 * np.sum(f1s_val[3])) ) / 10


            class_correct = np.array(class_correct)
            class_total = np.array(class_total)
            class_correctEb4 = np.array(class_correctEb4)
            class_totalEb4 = np.array(class_totalEb4)
            class_correctEb3 = np.array(class_correctEb3)
            class_totalEb3 = np.array(class_totalEb3)
            class_correctRes18 = np.array(class_correctRes18)
            class_totalRes18 = np.array(class_totalRes18)
            class_correct_val = np.array(class_correct_val)
            class_total_val = np.array(class_total_val)
            train_class_acc = class_correct / class_total
            train_class_accEb4 = class_correctEb4 / class_totalEb4
            train_class_accEb3 = class_correctEb3 / class_totalEb3
            train_class_accRes18 = class_correctRes18 / class_totalRes18
            
            train_class_acc_val = class_correct_val / class_total_val
            print("[*] val shape/min/max {}/{}/{}".format(x_val.shape, torch.min(x_val), torch.max(x_val)))
            print("[*] Eb4 Train acc # per class: 0={}, 1={}, 2={}, 3={}".format(train_class_accEb4[0], train_class_accEb4[1], train_class_accEb4[2], train_class_accEb4[3]))
            print("[*] Eb3 Train acc # per class: 0={}, 1={}, 2={}, 3={}".format(train_class_accEb3[0], train_class_accEb3[1], train_class_accEb3[2], train_class_accEb3[3]))
            print("[*] Res18 Train acc # per class: 0={}, 1={}, 2={}, 3={}".format(train_class_accRes18[0], train_class_accRes18[1], train_class_accRes18[2], train_class_accRes18[3]))
            print("[*] Train data # per class: 0={}, 1={}, 2={}, 3={}".format(class_total[0], class_total[1], class_total[2], class_total[3]))
            print("[*] Val data # per class: 0={}, 1={}, 2={}, 3={}".format(class_total_val[0], class_total_val[1], class_total_val[2], class_total_val[3]))
            print("[*] Train acc per class: 0={}, 1={}, 2={}, 3={}".format(train_class_acc[0], train_class_acc[1], train_class_acc[2], train_class_acc[3]))
            print("[*] Val acc per class: 0={}, 1={}, 2={}, 3={}".format(train_class_acc_val[0], train_class_acc_val[1], train_class_acc_val[2], train_class_acc_val[3]))
            print("[*] Train f1 per class: 0={}, 1={}, 2={}, 3={}".format(f1s_tr[0], f1s_tr[1], f1s_tr[2], f1s_tr[3]))
            print("[*] Val f1 per class: 0={}, 1={}, 2={}, 3={}".format(f1s_val[0], f1s_val[1], f1s_val[2], f1s_val[3]))

            acc = tp / a
            acc_val = tp_val / a_val
            print("  * loss = {}\n  * acc = {}\n  * loss_val = {}\n  * acc_val = {}\n * weighted f1 train = {}\n * weighted f1 val = {}".format(loss.item(), acc, loss_val.item(), acc_val, f1_weighted_tr, f1_weighted_val))
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=loss.item(), acc=acc, w_f1=f1_weighted_tr, val_loss=loss_val.item(), val_acc=acc_val, val_w_f1=f1_weighted_val)
            nsml.save(epoch)
            print('Training time for one epoch : %.1f\n' % (time.time() - t1))
        print('Total training time : %.1f' % (time.time() - t0))