import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class NeuralModule(nn.Module):

    def __init__(self, config = None):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def save_model(self, name=None, path="../../models"):
        if name is None:
            name = self.name
        os.makedirs(path, exist_ok=True)  
        path = f"{path}/{name}.pt"
        save_data = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'class_name': self.name
        }
        torch.save(save_data, path)
    
    def load_model(name, device, path="../../models"):
        path = f"{path}/{name}.pt"

        from attributor.TargetSourceAttributor import TAConfig
        torch.serialization.add_safe_globals([TAConfig])
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        #I know that it is not the best way
        #comment other models out because Aria don't have them in his branch
        #from attributor.SourceAttributorOuroboros import SourceAttributorOuroboros
        #from attributor.SourceAttributorTwins import SourceAttributorTwins
        #from attributor.LengthPredictorTranser import LengthPredictorTranser
        #from attributor.LengthPredictorConver import LengthPredictorConver
        #from attributor.SourceAttributorOuroborosHalt import SourceAttributorOuroborosHalt
        from attributor.TargetSourceAttributor import TargetSourceAttributor
        #from attributor.SourceAttributorTwinsBerts import SourceAttributorTwinsBerts
        model_class = locals()[checkpoint['class_name']]

        model = model_class(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    
    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
        