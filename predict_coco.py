import os
import sys
import pickle
import torch
from torchvision import transforms
from tqdm.autonotebook import tqdm
import json
import argparse
from recursion_schemes.recursion_schemes import ana_greedy,ana_beam,ana_mixed_beam
from bayesian_agents.joint_rsa import RSA
from rsa_utils.numpy_functions import uniform_vector, make_initial_prior
import logging
from PIL import Image

file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
sys.path.append(root_dir)

from utilities.data_utils import filename_from_id
from utilities.build_vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_img_dir(img_id, image_dir):

    img_path = os.path.join(
        image_dir, 'val2014/',
        filename_from_id(img_id, prefix='COCO_val2014_')
        )
    return img_path


def get_images_from_cluster(i, image_cluster, transform, image_dir):

    images = []
    # set up list containing data for all images in the current cluster
    for j in range(len(image_cluster[i])):

        image_id = image_cluster[i][j]
        img_dir = get_img_dir(image_id, image_dir)
        image = Image.open(img_dir).convert('RGB')
        image_tensor = transform(image).to(device).unsqueeze(0)

        images.append(image_tensor)

    return images


def rsa_decode_cluster(
    index, speaker_model, rat, image_cluster,
    transform, image_dir, beam=False, mixed=False,device=device
        ):

    image_tensors = get_images_from_cluster(
        index, image_cluster, transform, image_dir
        )

    # the model starts of assuming it's equally likely
    # any image is the intended referent
    initial_image_prior = uniform_vector(len(image_tensors))
    initial_rationality_prior = uniform_vector(1)
    initial_speaker_prior = uniform_vector(1)
    initial_world_prior = make_initial_prior(
        initial_image_prior, initial_rationality_prior, initial_speaker_prior)

    # set the possible images and rationalities
    speaker_model.initial_speakers.set_features(
        images=image_tensors, tf=False, rationalities=rat)

    if beam:
        if mixed:
            caption = ana_mixed_beam(
                speaker_model, target=0, speaker_rationality=0,
                speaker=0, start_from=list(""),
                initial_world_prior=initial_world_prior, no_progress_bar=True
                )
        else:
            caption = ana_beam(
                speaker_model, target=0, depth=1, speaker_rationality=0,
                speaker=0, start_from=list(""),
                initial_world_prior=initial_world_prior, no_progress_bar=True
                )
    else:
        if mixed:
            caption = ana_mixed_greedy(
                speaker_model, target=0, speaker_rationality=0,
                speaker=0, start_from=list(""),
                initial_world_prior=initial_world_prior, no_progress_bar=True
                )
        else:
            caption = ana_greedy(
                speaker_model, target=0, depth=1, speaker_rationality=0,
                speaker=0, start_from=list(""),
                initial_world_prior=initial_world_prior, no_progress_bar=True
                )

    out = {
        'image_id': int(image_cluster[index][0]),
        'caption': caption[0][0]  # [9:-5]  # slicing to remove <start> / <end>
    }

    return out


def rsa_decode_clusterloader(
    index, cluster_loader, speaker_model, rat, device=device
        ):

    cluster = cluster_loader[index]

    number_of_images = len(cluster)
    entry_ids = [c[0] for c in cluster]
    images = [c[2] for c in cluster]
    image_tensors = [image.to(device).unsqueeze(0) for image in images]

    # the model starts of assuming it's equally likely
    # any image is the intended referent
    initial_image_prior = uniform_vector(number_of_images)
    initial_rationality_prior = uniform_vector(1)
    initial_speaker_prior = uniform_vector(1)
    initial_world_prior = make_initial_prior(
        initial_image_prior, initial_rationality_prior, initial_speaker_prior)

    # first entry is target
    entry_id = entry_ids[0]
    entry = cluster_loader.dataset.df.loc[entry_id]

    out = {'image_id': int(entry.image_id)}

    # set the possible images and rationalities
    speaker_model.initial_speakers.set_features(
        images=image_tensors, tf=False, rationalities=rat)

    caption = ana_greedy(
        speaker_model, target=0, depth=1, speaker_rationality=0,
        speaker=0, start_from=list(""),
        initial_world_prior=initial_world_prior, no_progress_bar=True
    )

    out = {
        'image_id': int(entry.image_id),
        'caption': caption[0][0]  # [9:-5]  # slicing to remove <start> / <end>
    }

    return out


def main(args):

    #########
    # SETUP #
    #########

    logging.basicConfig(
        filename=args.out_dir + 'rsa_decoding_rat{}.log'.format(
            str(args.speaker_rat).replace('.', '-')
        ),
        level=logging.DEBUG)
    logging.info(('params:', args))

    print('----------------Setup----------------')

    # load vocab
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # define image transformation parameters
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # get cluster loader
    with open(args.cluster_path, 'rb') as f:
        image_clusters = pickle.load(f)

    # initialize speakers using the model parameters
    speaker_model = RSA(seg_type='char', vocabulary=vocab)
    speaker_model.initialize_speakers(model_path=args.model_path)

    # the rationality of the S1
    rat = [args.speaker_rat]

    #####################
    # GENERATE CAPTIONS #
    #####################

    print('----------------Decoding----------------')

    caps = []

    for i in tqdm(range(args.start_index,args.end_index)):
    #for i in tqdm(range(2)):
    #for i in tqdm(range(len(image_clusters))):

        c = rsa_decode_cluster(
            i, speaker_model, rat, image_clusters, transform, args.image_dir, beam=args.beam, mixed=args.mixed
            )
        logging.info(str(i) + ' ' + str(c))
        caps.append(c)

    #########################
    # write outputs to file #
    #########################

    print('----------------Saving File----------------')

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
        print('created dir', args.out_dir)

    d_name = 'greedy'
    if args.beam:
        d_name = 'beam'
    s_name = "single"
    if args.mixed:
        s_name = "mixed"


    file_path = os.path.join(
        args.out_dir,
        '{m_name}_{d_name}_{s_name}_rat{rat}_{start}_{end}.json'.format(
                m_name='adaptiveg_rsa_char',
                d_name=d_name,
                s_name=s_name,
                rat=str(args.speaker_rat).replace('.', '-'),
                start=args.start_index,
                end=args.end_index
            ).lower()
        )
    with open(file_path, 'w') as outfile:
        json.dump(caps, outfile)

    print('saved file to', file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', default='self',
                        help='To make it runnable in jupyter')
    parser.add_argument('--model_path', type=str,
                        default='/home/vu48pok/Dokumente/Projekte/diversity/diversity_fresh/data/models/speaker/adaptive-49.pkl',
                        help='path for trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='/home/vu48pok/Dokumente/Projekte/diversity/diversity_fresh/data/models/speaker/speaker2014_c.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='/home/vu48pok/.data/compling/data/corpora/external/MSCOCO/COCO/',
                        help='directory for resized training images')
    parser.add_argument('--out_dir', type=str,
                        default='/home/vu48pok/Dokumente/Projekte/diversity/diversity_fresh/data/models/speaker/',
                        help='output dir')
    parser.add_argument('--speaker_rat', type=float, default=5.0,
                        help='speaker rationality')
    parser.add_argument('--beam', action='store_true',
                        help='use beam search')
    parser.add_argument('--mixed', action='store_true',
                        help='use mixed (literal/rational) speakers')
    parser.add_argument('--splits_path', type=str,
                        default='/home/vu48pok/Dokumente/Projekte/diversity/diversity_fresh/data/karpathy_splits/',
                        help='path to splits')
    parser.add_argument('--captions_path', type=str,
                        default='/home/vu48pok/.data/compling/data/corpora/external/MSCOCO/COCO/',
                        help='path to captions')
    parser.add_argument('--cluster_path', type=str,
                        default='/home/vu48pok/Dokumente/Projekte/diversity/diversity_fresh/data/image_clusters_val.pkl',
                        help='path to image cluster file')
    parser.add_argument('--start_index', type=int,
                        default=0,
                        help='id of first cluster')
    parser.add_argument('--end_index', type=int,
                        default=5000,
                        help='id of last cluster')
    parser.add_argument('--split_name', type=str, default='val',
                        help='test or val split')

    args = parser.parse_args()

    print('----------------Settings----------------')
    print(args)

    main(args)
