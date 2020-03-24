import sys
import os
import numpy as np
from torchvision import transforms
import torch
from utils.config import sym_set
from utils.numpy_functions import softmax

# from train.image_captioning.char_model import EncoderCNN, DecoderRNN
model_path = '/home/simeon/Dokumente/Projekte/Diversity/NLGDiversity/code/models/vanilla'
sys.path.append(model_path)
from coco_model import EncoderCNN, DecoderRNN


class Model:

    def __init__(self, path, dictionaries):

        self.seg2idx, self.idx2seg = dictionaries
        self.path = path
        self.vocab_path = 'data/vocab.pkl'
        # self.encoder_path = TRAINED_MODEL_PATH + path + "-encoder-5-3000.pkl"
        # self.decoder_path = TRAINED_MODEL_PATH + path + "-decoder-5-3000.pkl"
        trained_model_dir = '/home/simeon/Dokumente/Projekte/Diversity/NLGDiversity/code/models/trained/'
        self.encoder_path = os.path.join(trained_model_dir ,'coco_captions-char-encoder-final.ckpt')
        self.decoder_path = os.path.join(trained_model_dir ,'coco_captions-char-decoder-final.ckpt')

        embed_size = 3  # 256
        hidden_size = 512
        num_layers = 1

        vocab_size = len(sym_set)  # i.e. 40, previously: 30

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        self.transform = transform
        # Load vocabulary wrapper

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Build Models
        self.encoder = EncoderCNN(embed_size)
        self.encoder.eval()  # evaluation mode (BN uses moving mean/variance)
        self.decoder = DecoderRNN(embed_size, hidden_size,
                                  vocab_size, num_layers)

        # Load the trained model parameters
        self.encoder.load_state_dict(torch.load(
            self.encoder_path, map_location=self.device))
        self.decoder.load_state_dict(torch.load(
            self.decoder_path, map_location=self.device))

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, world, state):

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

    def set_features(self, images, rationalities, tf):

        self.number_of_images = len(images)
        self.number_of_rationalities = len(rationalities)
        self.rationality_support = rationalities

        if tf:
            pass

        else:
            from utils.sample import to_var, load_image, load_image_from_path
            self.features = [self.encoder(
                to_var(load_image(url, self.transform), volatile=True)) for url in images]
            self.default_image = self.encoder(to_var(load_image_from_path(
                "data/default.jpg", self.transform), volatile=True))
