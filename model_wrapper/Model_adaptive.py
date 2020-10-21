import sys
import os
import numpy as np
from torchvision import transforms
import torch
from rsa_utils.numpy_functions import softmax
from rsa_utils.sample import load_image
import torch.nn.functional as F
from torch.autograd import Variable

file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(file_dir, os.pardir, os.pardir)
model_dir = os.path.join(root_dir, 'models')

sys.path.append(model_dir)
sys.path.append(model_dir+'/adaptiveg/')

from adaptive import Encoder2Decoder


class Model:

    """ wrap encoder and decoder models for rsa """

    def __init__(self, dictionaries, model_path):

        self.model_path = model_path

        self.seg2idx, self.idx2seg = dictionaries

        self.model = Encoder2Decoder(256, len(self.seg2idx), 512)
        self.model.load_state_dict(torch.load(
            self.model_path, map_location='cpu'
            ))
        self.model.cuda()
        self.model.eval()

        CROP_SIZE = 224
        transform = transforms.Compose([
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
                                 ])
        self.transform = transform

    def forward(self, world, state):

        V, v_g, captions = self.features[world.target]

        attention = []
        Beta = []

        states = None

        for seg in state.context_sentence:
            # decoding for current state of context sentence

            scores, states, atten_weights, beta = self.model.decoder(
                                                                V, v_g,
                                                                captions,
                                                                states
                                                                )

            scores = F.softmax(scores, dim=2)

            # set captions to the current segment for next step
            captions = Variable(
                torch.LongTensor(1, 1).fill_(self.seg2idx[seg]).cuda())

            attention.append(atten_weights)
            Beta.append(beta)

            # decoding for next step
        scores, states, atten_weights, beta = self.model.decoder(
                                                            V, v_g,
                                                            captions,
                                                            states
                                                            )
        scores = F.softmax(scores, dim=2)

        output_array = scores.squeeze(0).data.cpu().numpy()
        log_softmax_array = np.log(softmax(output_array))[0]

        return log_softmax_array

    def set_features(self, images, rationalities, format='image', tf=False):

        self.number_of_images = len(images)
        self.number_of_rationalities = len(rationalities)
        self.rationality_support = rationalities

        if format == 'url':
            images = [load_image(url, self.transform) for url in images]

        self.features = [
            self.model.init_sampler(i) for i in images
            ]
