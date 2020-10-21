import sys
import os
import numpy as np
from torchvision import transforms
import torch
from rsa_utils.numpy_functions import softmax
from rsa_utils.sample import to_var, load_image
import inspect

file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(file_dir, os.pardir, os.pardir)
model_dir = os.path.join(root_dir, 'models')

sys.path.append(model_dir)
import vanilla.coco_model as coco_model

class Model:

    """ wrap encoder and decoder models for rsa """

    def __init__(self, dictionaries, encoder_path, decoder_path):

        self.embed_size = 256
        self.hidden_size = 512
        self.num_layers = 1
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

        self.seg2idx, self.idx2seg = dictionaries

        vocab_size = len(self.seg2idx)

        CROP_SIZE = 224
        transform = transforms.Compose([
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        self.transform = transform

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Build Models
        self.encoder = coco_model.EncoderCNN(self.embed_size)
        self.encoder.eval()  # evaluation mode (BN uses moving mean/variance)
        self.decoder = coco_model.DecoderRNN(self.embed_size, self.hidden_size,
                                  vocab_size, self.num_layers)

        # Load the trained model parameters
        self.encoder.load_state_dict(torch.load(
            self.encoder_path, map_location=self.device))
        self.decoder.load_state_dict(torch.load(
            self.decoder_path, map_location=self.device))

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, world, state):
        #print(state.context_sentence)

        inputs = self.features[world.target].unsqueeze(1)

        states = None

        for seg in state.context_sentence:									  # maximum sampling length
            hiddens, states = self.decoder.lstm(
                inputs, states)		  # (batch_size, 1, hidden_size),

            outputs = self.decoder.linear(hiddens.squeeze(1))

            predicted = outputs.max(1)[1]

            predicted[0] = self.seg2idx[seg]
            inputs = self.decoder.embed(predicted)
            inputs = inputs.unsqueeze(1)		# (batch_size, vocab_size)

        hiddens, states = self.decoder.lstm(
            inputs, states)		  # (batch_size, 1, hidden_size),
        outputs = self.decoder.linear(hiddens.squeeze(1))
        output_array = outputs.squeeze(0).data.cpu().numpy()

        log_softmax_array = np.log(softmax(output_array))

        return log_softmax_array

    def set_features(self, images, rationalities, format='image', tf=False):
        print(inspect.currentframe())

        self.encoder.eval()
        self.decoder.eval()

        self.number_of_images = len(images)
        self.number_of_rationalities = len(rationalities)
        self.rationality_support = rationalities

        if format == 'url':
            images = [load_image(url, self.transform) for url in images]

        self.features = [
            self.encoder(to_var(i, volatile=True)) for i in images
            ]
