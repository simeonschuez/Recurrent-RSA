import numpy as np
from collections import defaultdict

max_sentence_length = 100
start_token = {"word":"<start>","char":'^'}
stop_token = {"word":"<end>","char":'$'}
pad_token = '&'
sym_set = list('&^$ abcdefghijklmnopqrstuvwxyz')

char_to_index = defaultdict(int)
for i,x in enumerate(sym_set):
    char_to_index[x] = i
index_to_char = defaultdict(lambda:'')
for i,x in enumerate(sym_set):
    index_to_char[i] = x


def vectorize_caption(sentence):
    if len(sentence) > 0 and sentence[-1] in list("!?."):
        sentence = sentence[:-1]
    sentence = start_token["char"] + sentence + stop_token["char"]
    sentence = list(sentence)
    while len(sentence) < max_sentence_length + 2:
        sentence.append(pad_token)

    caption_in = sentence[:-1]
    caption_out = sentence[1:]
    caption_in = np.asarray([char_to_index[x] for x in caption_in])
    caption_out = np.expand_dims(np.asarray(
        [char_to_index[x] for x in caption_out]), 0)
    one_hot = np.zeros((caption_out.shape[1], len(sym_set)))
    one_hot[np.arange(caption_out.shape[1]), caption_out] = 1
    caption_out = one_hot
    return caption_in, caption_out


class RSA_State:

    def __init__(
            self,
            initial_world_prior,
            listener_rationality=1.0
    ):

        self.context_sentence = np.expand_dims(
            np.expand_dims(vectorize_caption("")[0], 0), -1)

        # priors for the various dimensions of the world
        #  keep track of the dimensions of the prior
        self.dim = {"image": 0, "rationality": 1, "speaker": 2}

        # the priors at t>0 only matter if we aren't updating the prior at each step
        self.world_priors = np.asarray(
            [initial_world_prior for x in range(max_sentence_length + 1)])

        self.listener_rationality = listener_rationality

        # this is a bit confusing, isn't it
        self.timestep = 1

    def __hash__(self):
        return hash((self.timestep, self.listener_rationality, tuple(self.context_sentence)))
