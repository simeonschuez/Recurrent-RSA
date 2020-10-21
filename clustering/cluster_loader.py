import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(file_dir, os.pardir, os.pardir)

sys.path.append(root_dir)
from utilities.data_loader import COCOCaptionDataset, RefCOCODataset

class COCOClusters():
    """
    returns data for image clusters

    __getitem__ method:
        returns caption_ids, image_ids, and image data
        for cluster entries given a cluster id
    """
    def __init__(self, image_clusters,
                 decoding_level, list_IDs,
                 data_df, image_dir, vocab, transform):

        # initialize dataset wrapper
        self.dataset = COCOCaptionDataset(
            decoding_level=decoding_level, list_IDs=list_IDs, data_df=data_df,
            image_dir=image_dir, vocab=vocab, transform=transform
        )
        self.image_clusters = image_clusters

    def __len__(self):
        # number of images in loader
        return len(self.image_clusters)

    def __getitem__(self, index):
        """
            ::input:: index for cluster in self.image_clusters
            ::output:: caption_ids, image_ids and image data for that cluster
        """
        # retrieve indices in id list, caption ids
        # and image ids for entries in cluster
        id_list_idx, caption_ids, image_ids = self.cluster_ids(index)
        # retrieve image data from self.dataset for the given indices
        images = [self.dataset[i][0] for i in id_list_idx]

        return list(zip(caption_ids, image_ids, images))

    def cluster_ids(self, i):
        df = self.dataset.df
        # select cluster from image_clusters
        image_ids = self.image_clusters[i]

        # retrieve caption_ids from data frame, given the image ids:
        # 1) set caption_id as column (not index)
        # 2) select first entry for each image id
        # 3) restore order from image_ids list
        # 4) get caption ids as list
        caption_ids = df.loc[df.image_id.isin(image_ids)]\
            .reset_index()\
            .groupby('image_id').agg('first')\
            .loc[image_ids]\
            .id.to_list()

        # get position from caption_ids in dataset id list
        # (used for retrieving entries from dataset wrapper)
        ids_idx = [
            self.dataset.list_IDs.index(i)
            for i in caption_ids
        ]

        return(ids_idx, caption_ids, image_ids)


class RefCOCOClusters():
    """
    returns data for targets and distractors in refcoco(+) images

    __getitem__ method:
        returns caption_ids, image_ids, and image data
        for target and distractors given a target index
    image_entities method:
        returns caption_ids, image_ids, and image data
        for objects in an image given an image id
    """

    def __init__(self, decoding_level, list_IDs,
                 data_df, image_dir, vocab, transform):

        # initialize dataset wrapper
        self.dataset = RefCOCODataset(
            decoding_level=decoding_level, list_IDs=list_IDs, data_df=data_df,
            image_dir=image_dir, vocab=vocab, transform=transform
        )

    def __len__(self):
        # number of objects in loader
        return(len(self.dataset))

    def __getitem__(self, index):
        """
            ::input:: index for cluster in self.image_clusters
            ::output:: sent_ids, image_ids and image data for that cluster
        """
        # retrieve indices in id list, caption ids and image ids for entries in cluster
        id_list_idx, sent_ids, ann_ids, image_id = self.get_distractors(index)
        # retrieve image data from self.dataset for the given indices
        image_data = [self.dataset[i] for i in id_list_idx]
        # unpack image data
        images = [entry[0] for entry in image_data]
        pos_infs = [entry[2] for entry in image_data]

        return list(zip(sent_ids, ann_ids, images, pos_infs))

    def image_entities(self, img_id):
        """
            ::input:: image_id
            ::output:: sent_ids and image data for objects in image
        """
        df = self.dataset.df
        df = df.reset_index()
        # get entries in image
        image_entries = df.loc[df.image_id == img_id]
        # get first sent_id for every object in image
        sent_ids = image_entries.reset_index()\
            .groupby('ann_id')\
            .agg('first')\
            .sent_id.to_list()

        # get position from sent_ids in dataset id list
        # (used for retrieving entries from dataset wrapper)
        ids_idx = [
            i for i, c_id in enumerate(self.dataset.list_IDs)
            if c_id in sent_ids
            ]

        # retrieve image data from self.dataset for the given indices
        image_data = [self.dataset[i] for i in ids_idx]
        # unpack image data
        images = [entry[0] for entry in image_data]
        pos_infs = [entry[2] for entry in image_data]

        return list(zip(sent_ids, images, pos_infs))

    def get_distractors(self, i):
        """
            ::input:: id for target object
            ::output:: sent_ids and positions in dataset id list
                       for target and distractors in same image
        """

        df = self.dataset.df
        # get target entry
        target = df.loc[self.dataset.list_IDs[i]]
        target_sent_id = target.name
        target_ann_id = target.ann_id
        image_id = target.image_id

        # get other entries in same image
        other_entries = df.loc[df.image_id == target.image_id].loc[df.ann_id != target.ann_id]

        # get unique objects
        distractors = other_entries\
            .reset_index()\
            .groupby('ann_id').agg('first')\
            .reset_index().set_index('sent_id')

        distractors_sent_ids = distractors.index.to_list()
        distractors_ann_ids = distractors.ann_id.to_list()

        # combine target and distractor ids
        sent_ids = [target_sent_id] + distractors_sent_ids
        ann_ids = [target_ann_id] + distractors_ann_ids

        # get position from sent_ids in dataset id list
        # (used for retrieving entries from dataset wrapper)
        ids_idx = [
            self.dataset.list_IDs.index(i)
            for i in sent_ids
        ]

        return(ids_idx, sent_ids, ann_ids, image_id)


def iterate_targets(lst):
    """
    set every entry in lst as target once
    use other entries as distractors
    """
    for i in range(len(lst)):
        # get target + distractors at current step
        target = lst[i]
        distractors = lst[:i] + lst[i+1:]

        # return items in current order
        yield [target] + distractors
