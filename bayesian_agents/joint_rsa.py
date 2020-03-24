# import time
import itertools
import scipy
import scipy.stats
import numpy as np
from utils.config import index_to_char, char_to_index
from bayesian_agents.rsaWorld import RSA_World
from train.Model import Model


class RSA:

    """
    RSA Class:
    :contains:
        - initialize_speakers: Setting up models for initial speakers + prior
        - speaker function: posterior distribution over tokens
        - listener function
        - util functions (memoize, flush_cache)
    """

    def __init__(
            self,
            seg_type,
            tf,
    ):

        self.tf = tf
        self.seg_type = seg_type
        self.char = self.seg_type = "char"

        # caches for memoization
        self._speaker_cache = {}
        self._listener_cache = {}
        self._speaker_prior_cache = {}

        if self.char:
            self.idx2seg = index_to_char
            self.seg2idx = char_to_index

    def initialize_speakers(self, paths):

        """ initialize neural models for initial speakers + prior """

        self.initial_speakers = [Model(path=path,
                                       dictionaries=(self.seg2idx, self.idx2seg)) for path in paths]
        self.speaker_prior = Model(path="lang_mod",
                                   dictionaries=(self.seg2idx, self.idx2seg))

    def flush_cache(self):

        self._speaker_cache = {}
        self._listener_cache = {}
        self._speaker_prior_cache = {}

    def memoize_speaker(f):
        def helper(self, state, world, depth):

            hashable_args = state, world, depth

            if hashable_args not in self._speaker_cache:
                self._speaker_cache[hashable_args] = f(
                    self, state, world, depth)
            # else: print("cached")
            return self._speaker_cache[hashable_args]
        return helper

    def memoize_listener(f):
        def helper(self, state, utterance, depth):

            hashable_args = state, utterance, depth

            if hashable_args not in self._listener_cache:
                self._listener_cache[hashable_args] = f(
                    self, state, utterance, depth)

            return self._listener_cache[hashable_args]
        return helper

    @memoize_speaker
    def speaker(self, state, world, depth):
        """
        :input: RSA State , RSA world, depth
        :returns: posterior for tokens
        """

        if depth == 0:
            return self.initial_speakers[world.speaker].forward(state=state, world=world)

        else:
            prior = self.speaker(state, world, depth=0)

        if depth == 1:

            scores = []
            for k in range(prior.shape[0]):
                out = self.listener(state=state, utterance=k, depth=depth - 1)

                scores.append(
                    out[world.target, world.rationality, world.speaker])

            scores = np.asarray(scores)
            scores = scores * \
                (self.initial_speakers[world.speaker]
                 .rationality_support[world.rationality])
            posterior = (scores + prior) - scipy.special.logsumexp(scores + prior)

            return posterior

        elif depth == 2:

            scores = []
            for k in range(prior.shape[0]):

                out = self.listener(state=state, utterance=k, depth=depth - 1)
                scores.append(
                    out[world.target, world.rationality, world.speaker])

            scores = np.asarray(scores)
            posterior = (scores + prior) - scipy.special.logsumexp(scores + prior)

            return posterior

    @memoize_listener
    def listener(self, state, utterance, depth):

        # base case listener is either neurally trained, or inferred from neural s0, given the state's current prior on images

        world_prior = state.world_priors[state.timestep - 1]
        scores = np.zeros((world_prior.shape))
        for n_tuple in itertools.product(*[list(range(x)) for x in world_prior.shape]):

            world = RSA_World(target=n_tuple[state.dim["image"]],
                              rationality=n_tuple[state.dim["rationality"]], speaker=n_tuple[state.dim["speaker"]])

            # NOTE THAT NOT DEPTH-1 HERE
            out = self.speaker(state=state, world=world, depth=depth)

            scores[n_tuple] = out[utterance]

        scores = scores * state.listener_rationality
        world_posterior = (scores + world_prior) - \
            scipy.special.logsumexp(scores + world_prior)
        return world_posterior

    def listener_simple(self, state, utterance, depth):

        # base case listener is either neurally trained,
        # or inferred from neural s0, given the state's current prior on images

        world_prior = state.world_priors[state.timestep - 1]
        assert world_prior.shape == (2, 1, 1)
        print("world prior", np.exp(world_prior))

        # I could write: itertools product axes
        scores = np.zeros((2, 1, 1))
        for i in range(2):
            world = RSA_World(target=i, rationality=0, speaker=0)

            # NOTE THAT NOT DEPTH-1 HERE
            out = self.speaker(state=state, world=world, depth=depth)

            scores[i] = out[utterance]

        scores = scores * state.listener_rationality
        world_posterior = (scores + world_prior) - \
            scipy.special.logsumexp(scores + world_prior)

        return world_posterior
