import torch
import torchvision.models.vgg as vgg
from collections import namedtuple


LossOutput = namedtuple(
   "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])

class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        # kwargs = {}
        # kwargs['init_weights'] = False
        # model = vgg.VGG(vgg.make_layers(vgg.cfgs['E'], batch_norm=False), **kwargs)
        # state_dict = torch.load("vgg_model/vgg19-dcbb9e9d.pth", map_location='cuda:0')
        # model.load_state_dict(state_dict)
        # self.vgg_layers = model.features
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        #

        self.layer_name_mapping = {
            '1': "relu1",
            '6': "relu2",
            '11': "relu3",
            '20': "relu4",
            '29': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name == '29':
                break
        return LossOutput(**output)


class Perceptual_loss(torch.nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()

        #self.model = models_lpf.resnet50(filter_size = 5)
        #self.model.load_state_dict(torch.load('/data/wmy/NR/models/resnet50_lpf5.pth.tar')['state_dict'])
        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()
        self.loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, x, target):
        x_feature = self.model(x)
        target_feature = self.model(target)

        feature_loss = (self.loss(x_feature.relu1,target_feature.relu1)
                        +self.loss(x_feature.relu2,target_feature.relu2)
                        +self.loss(x_feature.relu3,target_feature.relu3)
                        +self.loss(x_feature.relu4,target_feature.relu4)
                        +self.loss(x_feature.relu5,target_feature.relu5) ) /5.0

        return feature_loss






