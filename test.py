## import libraries for training
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
warnings.filterwarnings('ignore')

import my_models
from sklearn.metrics import precision_score, recall_score, f1_score

# Validating the model
def evaluate(val_loader,model):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    acc = AverageMeter()
    # for P, R and F1
    predictions = []
    true_labels = []

    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)

            predictions.extend(preds.argmax(dim=1).cpu().numpy())  # get arg of top prob for every test case
            true_labels.extend(label.cpu().numpy())
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            acc.update(valid_acc1, img.size(0))

    print("\nRESULTS: -")
    print("mAP: ", map.avg)
    print("Accuracy: ", acc.avg, " %")
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    print("\nPrecision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    return map.avg

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

######################## load file and get splits #############################
print('reading test file')
test_files = pd.read_csv("test.csv")
print('Creating test dataloader')
test_gen = knifeDataset(test_files,mode="val")
test_loader = DataLoader(test_gen,batch_size=64,shuffle=False,pin_memory=True,num_workers=8)

print('loading trained model: ')
model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=config.n_classes)
# 20 ep - 0.6742, 30 ep - 
# model = timm.create_model('resnet34', pretrained=True,num_classes=config.n_classes)
# 0.6742
# model = my_models.MyNet()
# 0.6840
# model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=192)

filepath = 'logs/trivwide/Knife-Effb0-E20.pt'
model.load_state_dict(torch.load(filepath))
# OR
# checkpoint = torch.load(filepath)
# model.load_state_dict(checkpoint['model_state_dict'])

# print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Training #################################
print('Evaluating trained model')
map = evaluate(test_loader,model)
# print("mAP =",map)
    
   
