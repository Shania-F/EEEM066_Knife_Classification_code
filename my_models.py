import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import timm


class MyNet(nn.Module):
    '''
    Googlenet with an additional fc layer to take it from 1000 to 192
    '''
    def __init__(self, droprate=0.5, pretrained=True):
        super(MyNet, self).__init__()
        self.droprate = droprate

        self.feature_extractor = timm.create_model('inception_v3', pretrained=True)
        self.fc_last = nn.Linear(1000, 192)


    def forward(self, x):
        # 1 x 3 x 229 x 299
        x = self.feature_extractor(x)
        # 1 x 1000
        x = F.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        logits = self.fc_last(x)
        # 1 x 192
        return logits


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyNet()
    model.to(device)
    summary(model, (3, 224, 224))
