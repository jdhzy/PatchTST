import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    ##########Originally from Exp_Basic.py##########
    # def _acquire_device(self):
    #     if self.args.use_gpu:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #             self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device
    ##########End of Exp_Basic.py##########
    def _acquire_device(self):
        device = self.args.device
        if device.type == 'cuda':
            print(f'Use GPU: {device}')
        elif device.type == 'mps':
            print('Use Apple Metal (MPS) backend')
        else:
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self, setting):
        pass

    def test(self):
        pass
