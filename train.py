## import libraries for training
import sys
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

from torch.utils.tensorboard import SummaryWriter
import my_models

## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
log.open("logs/%s_log_train.txt")
log.write(f"DefaultConfigs(n_classes={config.n_classes}, img_weight={config.img_weight}, img_height={config.img_height}, "
               f"batch_size={config.batch_size}, epochs={config.epochs}, learning_rate={config.learning_rate})")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|---------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start):
    losses = AverageMeter()
    model.train()
    model.training=True
    for i,(images,target,fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)  # lets it transfer asynchronously, improves performance
        label = target.cuda(non_blocking=True)

        # context manager
        # the operations within its scope are performed with reduced precision (FP16) where possible
        # improves performance
        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(),images.size(0))  # average loss over batch of 16
        scaler.scale(loss).backward()  # CALC GRADIENTS
        scaler.step(optimizer)   # UPDATE WEIGHTS
        scaler.update()  # UPDATE SCALER WEIGHTS (scaler improves performance)
        optimizer.zero_grad()
        scheduler.step()

        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % (\
                "train", i, epoch,losses.avg,valid_accuracy[0],time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]

# Validating the model
def evaluate(val_loader,model,criterion,epoch,train_loss,start):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)  # pytorch method to apply softmax on tensor along dim 1

            loss = criterion(logits, label)
            losses.update(loss.item(), images.size(0))
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], map.avg,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
    return [map.avg, losses.avg]

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        # returns the top k probabilites predicted (value) and the index (top)
        # obviously top 1 is the prediction
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        # tensor where the correct class is True
        # [ True, False, False, False, False] -> correct top 1
        # [False, False, True, False, False] -> correct top 3
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        # correct[0] -> top 1 acc, remaining is cumulative top 5
        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

######################## load file and get splits #############################
train_imlist = pd.read_csv("train.csv")
train_gen = knifeDataset(train_imlist,mode="train")
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=8)
val_imlist = pd.read_csv("test.csv")
val_gen = knifeDataset(val_imlist,mode="val")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=8)

## Loading the model to run
# model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=config.n_classes)
# model = timm.create_model('resnet34', pretrained=True, num_classes=config.n_classes)
# model = my_models.MyNet()
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=192)

log.write('Using model: vit_small\n')

# TODO if checkpoint exists, load
# if os.path.exists("./Knife-Effb0-E20.pt"):
#   print("Loading saved checkpoint")
#   model.load_state_dict(torch.load('./Knife-Effb0-E10.pt'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Parameters #################################
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()

############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()  # Assuming you have scaler defined somewhere
# used to keep gradients in a good range, Mixed Precision (AMP)
start = timer()
writer = SummaryWriter(log_dir='logs')

#train
for epoch in range(0,config.epochs):
    lr = get_learning_rate(optimizer)  # leftover code
    train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start)
    val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,start)

    # Tensorboard
    writer.add_scalars('loss/trainval', {'train': train_metrics[0], 'validation': val_metrics[1]}, epoch + 1)

    # Saving the model
    if (epoch + 1)%10 == 0:
        filename = "logs/Knife-Effb0-E" + str(epoch + 1)+  ".pt"
        torch.save(model.state_dict(), filename)
        # TODO save all so we can reload training
    
writer.close()
