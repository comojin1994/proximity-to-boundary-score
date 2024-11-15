from models.backbones.eegnet import EEGNet
from models.backbones.deepconvnet import DeepConvNet
from models.backbones.shallowconvnet import ShallowConvNet
from models.litmodels.litmodellinear import LitModelLinear, LitModelWeightedLinear
from easydict import EasyDict

backbone_dict = {
    "eegnet": EEGNet,
    "deepconvnet": DeepConvNet,
    "shallowconvnet": ShallowConvNet,
}

litmodel_dict = {
    "base": LitModelLinear,
    "weighted": LitModelWeightedLinear,
}


class ModelMaker:
    def __init__(self, model_name, litmodel_name):

        self.encoder = backbone_dict[model_name]
        self.litmodel = litmodel_dict[litmodel_name]

    def load_model(self, args: EasyDict, **kwargs):
        encoder = self.encoder(args, **kwargs)
        litmodel = self.litmodel(encoder, args)

        return litmodel
