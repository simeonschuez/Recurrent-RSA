from tqdm.autonotebook import tqdm
import numpy as np
import time
import math
import copy
from rsa_utils.config import stop_token, sym_set, max_sentence_length, index_to_char
from bayesian_agents.rsaState import RSA_State
from bayesian_agents.rsaWorld import RSA_World

###
"""
Recursion schemes for the RSA
"""
###

# Sina: set pass prior to FALSE! This is much faster
def ana_greedy(
        rsa, initial_world_prior, speaker_rationality, speaker, target,
        pass_prior=False, listener_rationality=1.0, depth=0, start_from=[],
        start_token='<start>', end_token='<end>', no_progress_bar=False
        ):

    """
    speaker_rationality,listener_rationality:

            see speaker and listener code for what they do: control strength of conditioning

    depth:

            the number of levels of RSA: depth 0 uses listeral speaker to unroll, depth n uses speaker n to unroll, and listener n to update at each step

    start_from:

            a partial caption you start the unrolling from

    img_prior:

            a prior on the world to start with

    :Paper:
    To perform greedy unrolling [...] for either S0 or S1, we initialize the state as a partial caption pc0 consisting of only the start token, and a uniform prior over the images ip0.
    Then, for t > 0, we use our incremental speaker model S0 or S1 to generate a distribution over the subsequent character S t (u|w, ipt, pct), and add the character u with highest probability density to pct , giving us pct+1.
    We then run our listener model L1 on u, to obtain a distribution ipt+1 = Lt 1(w|u, ipt , pct ) over images that the L0 can use at the next timestep.
    """

    # this RSA passes along a state: see rsa_state
    # contains initial world priors (default: uniform priors for all tokens)
    state = RSA_State(initial_world_prior,
                      listener_rationality=listener_rationality)

    # initialize partial context caption with start symbol
    context_sentence = [start_token] + start_from
    state.context_sentence = context_sentence

    world = RSA_World(
        target=target, rationality=speaker_rationality, speaker=speaker)

    probs = []

    # perform greedy unrolling
    for timestep in tqdm(
        range(len(start_from) + 1, max_sentence_length),
        disable=no_progress_bar
            ):

        # update state.timestep
        state.timestep = timestep
        # get probability distribution over possible next tokens from (pragmatic) speaker
        s = rsa.speaker(state=state, world=world, depth=depth)
        # next token
        segment = np.argmax(s)
        # probability for next token
        prob = np.max(s)
        probs.append(prob)

        if pass_prior:
            # update world_priors with rsa.listener:
            # informed decision from rational speaker for future timesteps
            # (instead of uniform prior over possible targets)
            l = rsa.listener(state=state, utterance=segment, depth=depth)
            state.world_priors[state.timestep] = l

        # add next token to context_sentence
        state.context_sentence += [rsa.idx2seg[segment]]
        # break if stop token has been reached
        if (rsa.idx2seg[segment] == end_token):  # stop_token[rsa.seg_type]):
            break

    summed_probs = np.sum(np.asarray(probs))

    world_posterior = state.world_priors[:state.timestep+1][:5]

    return [("".join(state.context_sentence), summed_probs)]


def ana_mixed_greedy(
        rsa, initial_world_prior, speaker_rationality, speaker, target,
        pass_prior=False, listener_rationality=1.0, start_from=[],
        start_token='<start>', end_token='<end>', no_progress_bar=False
        ):

    """
    speaker_rationality,listener_rationality:

            see speaker and listener code for what they do: control strength of conditioning

    depth:

            the number of levels of RSA: depth 0 uses listeral speaker to unroll, depth n uses speaker n to unroll, and listener n to update at each step

    start_from:

            a partial caption you start the unrolling from

    img_prior:

            a prior on the world to start with

    :Paper:
    To perform greedy unrolling [...] for either S0 or S1, we initialize the state as a partial caption pc0 consisting of only the start token, and a uniform prior over the images ip0.
    Then, for t > 0, we use our incremental speaker model S0 or S1 to generate a distribution over the subsequent character S t (u|w, ipt, pct), and add the character u with highest probability density to pct , giving us pct+1.
    We then run our listener model L1 on u, to obtain a distribution ipt+1 = Lt 1(w|u, ipt , pct ) over images that the L0 can use at the next timestep.
    """

    # this RSA passes along a state: see rsa_state
    # contains initial world priors (default: uniform priors for all tokens)
    state = RSA_State(initial_world_prior,
                      listener_rationality=listener_rationality)

    # initialize partial context caption with start symbol
    context_sentence = [start_token] + start_from
    state.context_sentence = context_sentence

    world = RSA_World(
        target=target, rationality=speaker_rationality, speaker=speaker)

    probs = []

    # perform greedy unrolling
    for timestep in tqdm(
        range(len(start_from) + 1, max_sentence_length),
        disable=no_progress_bar
            ):

        # update state.timestep
        state.timestep = timestep

        if state.timestep > 1:

            seg = sent[-1]

            if seg == ' ':
                depth = 1
            else:
                depth = 0

        else:
            depth = 1

        # get probability distribution over possible next tokens from (pragmatic) speaker
        s = rsa.speaker(state=state, world=world, depth=depth)
        # next token
        segment = np.argmax(s)
        # probability for next token
        prob = np.max(s)
        probs.append(prob)

        if pass_prior:
            # update world_priors with rsa.listener:
            # informed decision from rational speaker for future timesteps
            # (instead of uniform prior over possible targets)
            l = rsa.listener(state=state, utterance=segment, depth=depth)
            state.world_priors[state.timestep] = l

        # add next token to context_sentence
        state.context_sentence += [rsa.idx2seg[segment]]
        # break if stop token has been reached
        if (rsa.idx2seg[segment] == end_token):  # stop_token[rsa.seg_type]):
            break

    summed_probs = np.sum(np.asarray(probs))

    world_posterior = state.world_priors[:state.timestep+1][:5]

    return [("".join(state.context_sentence), summed_probs)]

# But within the n-th order ethno-metapragmatic perspective, this creative indexical effect is the motivated realization, or performable execution, of an already constituted framework of semiotic value.

# Sina: set pass prior to FALSE! This is much faster
def ana_beam(rsa, initial_world_prior, speaker_rationality, target, speaker, pass_prior=False, listener_rationality=1.0, depth=0, start_from=[], beam_width=5, cut_rate=1, decay_rate=0.0, beam_decay=0, no_progress_bar=False):
    """
    speaker_rationality,listener_rationality:
            see speaker and listener code for what they do: control strength of conditioning

    depth:
            the number of levels of RSA: depth 0 uses listeral speaker to unroll, depth n uses speaker n to unroll, and listener n to update at each step

    start_from:
            a partial caption you start the unrolling from

    img_prior:
            a prior on the world to start with

    which_image:
            which of the images in the prior should be targeted?

    beam width:
            width beam is cut down to every cut_rate iterations of the unrolling

    cut_rate:
            how often beam is cut down to beam_width

    beam_decay:
            amount by which beam_width is lessened after each iteration

    decay_rate:
            a multiplier that makes later decisions in the unrolling matter less: 0.0 does no decay. negative decay makes start matter more

    """
    print("Hello this is beam search")

    # initialize state object
    # (containing initial - uniform - world priors for t0)
    # default: start from empty token list
    state = RSA_State(initial_world_prior,
                      listener_rationality=listener_rationality)
    context_sentence = start_from
    state.context_sentence = context_sentence

    # intitialize world object
    world = RSA_World(
        target=target, rationality=speaker_rationality, speaker=speaker)

    context_sentence = start_from
    state.context_sentence = context_sentence

    # initial probability value
    sent_worldprior_prob = [(state.context_sentence, state.world_priors, 0.0)]

    final_sentences = []

    toc = time.time()
    # iterate through individual time steps / tokens to be produced
    for timestep in tqdm(range(len(start_from) + 1, max_sentence_length), disable=no_progress_bar):
        #print("this is timestep",timestep)
        #print(len(sent_worldprior_prob))

        state.timestep = timestep

        new_sent_worldprior_prob = []
        # iterate through previous beams
        for sent, worldpriors, old_prob in sent_worldprior_prob:
            #print("\t",sent,old_prob)

            state.world_priors = worldpriors

            if state.timestep > 1:
                # get listener prior over targets
                # if depth > 0 and pass_prior is set

                state.context_sentence = sent[:-1]
                seg = sent[-1]

                if depth > 0 and pass_prior:

                    l = rsa.listener(
                        state=state, utterance=rsa.seg2idx[seg], depth=depth)
                    state.world_priors[state.timestep - 1] = copy.deepcopy(l)

            state.context_sentence = sent

            # get probability distribution over possible next tokens from speaker
            s = rsa.speaker(state=state, world=world, depth=depth)

            #print(s)
            top_segs = np.argsort(s)[::-1][:beam_width]

            # iterate through possible next tokens
            for seg in top_segs:
                prob = s[seg]
                # add new segment to existing sentence

                new_sentence = copy.deepcopy(sent)

                # conditional to deal with captions longer than max sentence length
                # if state.timestep<max_sentence_length+1:
                new_sentence += [rsa.idx2seg[seg]]
                # else: new_sentence = np.expand_dims(np.expand_dims(np.concat([np.squeeze(new_sentence)[:-1],[seg]],axis=0),0),-1)

                state.context_sentence = new_sentence

                # calculate new probability for resulting sentence
                new_prob = (
                    prob * (1 / math.pow(state.timestep, decay_rate))) + old_prob

                new_sent_worldprior_prob.append(
                    (new_sentence, worldpriors, new_prob))

        rsa.flush_cache()
        # sort possible next sentences by probability (descending)
        sent_worldprior_prob = sorted(
            new_sent_worldprior_prob, key=lambda x: x[-1], reverse=True)
        sent_worldprior_prob = sent_worldprior_prob[:beam_width]

        if len(sent_worldprior_prob) > 1:

            top_sent = sent_worldprior_prob[0][0]
            #print("top sent",top_sent)

            if top_sent[-1] == '<end>':
                output = []
                print(top_sent)
                for sent, worldprior, prob in sent_worldprior_prob:
                # add sentences with stop token to final_sentences
                    final_sentence = copy.deepcopy(sent)
                    output.append(("".join(final_sentence), prob))
                return output

    output = []
    for sent, worldprior, prob in sent_worldprior_prob:
        # add sentences with stop token to final_sentences
        final_sentence = copy.deepcopy(sent)
        output.append(("".join(final_sentence), prob))
    return output




        # cut off least probable sentence candidates
        # if timestep matches cut_rate
        #if state.timestep % cut_rate == 0:
            # cut down to size
            #print("hello cut rate")
            #sent_worldprior_prob = sent_worldprior_prob[:beam_width]
            #new_sent_worldprior_prob = []

            #for sent, worldprior, prob in sent_worldprior_prob:
                # add sentences with stop token to final_sentences
                #if sent[-1] == stop_token[rsa.seg_type]:
                    #final_sentence = copy.deepcopy(sent)
                    #final_sentences.append((final_sentence, prob))
                # add unfinished sentences to new_sent_worldprior_prob
                #else:
                    #new_triple = copy.deepcopy((sent, worldprior, prob))
                    #new_sent_worldprior_prob.append(new_triple)

            # update sent_worldprior_prob for next iteration
            #sent_worldprior_prob = new_sent_worldprior_prob

            # return sentence if 50 complete sentences were gathered
            #if len(final_sentences) >= 50:
                #sentences = sorted(
                #    final_sentences, key=lambda x: x[-1], reverse=True)
                #output = []
                #for i, (sent, prob) in enumerate(sentences):

                    #output.append(("".join(sent), prob))

                #return output

        #if beam_decay < beam_width:
            #beam_width -= beam_decay
        # print("decayed beam width by "+str(beam_decay)+"; beam_width now: "+str(beam_width))

    # indentation?
    #else:
    #    sentences = sorted(final_sentences, key=lambda x: x[-1], reverse=True)

    #    output = []
        # print(sentences)
    #    for i, (sent, prob) in enumerate(sentences):

    #        output.append(("".join(sent), prob))

    #    return output
def ana_mixed_beam(rsa, initial_world_prior, speaker_rationality, target, speaker, pass_prior=False, listener_rationality=1.0, start_from=[], beam_width=5, cut_rate=1, decay_rate=0.0, beam_decay=0, no_progress_bar=False):
    """
    speaker_rationality,listener_rationality:
            see speaker and listener code for what they do: control strength of conditioning

    depth:
            the number of levels of RSA: depth 0 uses listeral speaker to unroll, depth n uses speaker n to unroll, and listener n to update at each step

    start_from:
            a partial caption you start the unrolling from

    img_prior:
            a prior on the world to start with

    which_image:
            which of the images in the prior should be targeted?

    beam width:
            width beam is cut down to every cut_rate iterations of the unrolling

    cut_rate:
            how often beam is cut down to beam_width

    beam_decay:
            amount by which beam_width is lessened after each iteration

    decay_rate:
            a multiplier that makes later decisions in the unrolling matter less: 0.0 does no decay. negative decay makes start matter more

    """
    print("Hello this is mixed beam search")

    # initialize state object
    # (containing initial - uniform - world priors for t0)
    # default: start from empty token list
    state = RSA_State(initial_world_prior,
                      listener_rationality=listener_rationality)
    context_sentence = start_from
    state.context_sentence = context_sentence

    # intitialize world object
    world = RSA_World(
        target=target, rationality=speaker_rationality, speaker=speaker)

    context_sentence = start_from
    state.context_sentence = context_sentence

    # initial probability value
    sent_worldprior_prob = [(state.context_sentence, state.world_priors, 0.0)]

    final_sentences = []

    toc = time.time()
    # iterate through individual time steps / tokens to be produced
    for timestep in tqdm(range(len(start_from) + 1, max_sentence_length), disable=no_progress_bar):
        #print("this is timestep",timestep)
        #print(len(sent_worldprior_prob))

        state.timestep = timestep

        new_sent_worldprior_prob = []
        # iterate through previous beams
        for sent, worldpriors, old_prob in sent_worldprior_prob:
            #print("\t",sent,old_prob)

            state.world_priors = worldpriors

            if state.timestep > 1:

                seg = sent[-1]

                if seg == ' ':
                    depth = 1
                else:
                    depth = 0

            else:
                depth = 1

            #print("depth",depth)

            state.context_sentence = sent

            # get probability distribution over possible next tokens from speaker
            s = rsa.speaker(state=state, world=world, depth=depth)

            #print(s)
            top_segs = np.argsort(s)[::-1][:beam_width]

            # iterate through possible next tokens
            for seg in top_segs:
                prob = s[seg]
                # add new segment to existing sentence

                new_sentence = copy.deepcopy(sent)

                # conditional to deal with captions longer than max sentence length
                # if state.timestep<max_sentence_length+1:
                new_sentence += [rsa.idx2seg[seg]]
                # else: new_sentence = np.expand_dims(np.expand_dims(np.concat([np.squeeze(new_sentence)[:-1],[seg]],axis=0),0),-1)

                state.context_sentence = new_sentence

                # calculate new probability for resulting sentence
                new_prob = (
                    prob * (1 / math.pow(state.timestep, decay_rate))) + old_prob

                new_sent_worldprior_prob.append(
                    (new_sentence, worldpriors, new_prob))

        rsa.flush_cache()
        # sort possible next sentences by probability (descending)
        sent_worldprior_prob = sorted(
            new_sent_worldprior_prob, key=lambda x: x[-1], reverse=True)
        sent_worldprior_prob = sent_worldprior_prob[:beam_width]

        if len(sent_worldprior_prob) > 1:

            top_sent = sent_worldprior_prob[0][0]
            #print("top sent",top_sent)

            if top_sent[-1] == '<end>':
                output = []
                print(top_sent)
                for sent, worldprior, prob in sent_worldprior_prob:
                # add sentences with stop token to final_sentences
                    final_sentence = copy.deepcopy(sent)
                    output.append(("".join(final_sentence), prob))
                return output

    output = []
    for sent, worldprior, prob in sent_worldprior_prob:
        # add sentences with stop token to final_sentences
        final_sentence = copy.deepcopy(sent)
        output.append(("".join(final_sentence), prob))
    return output
