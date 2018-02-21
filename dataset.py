import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary

class DataSet(object):
    def __init__(self,
                 image_ids,
                 image_files,
                 batch_size,
                 word_idxs=None,
                 masks=None,
                 is_train=False,
                 shuffle=False):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)
        self.word_idxs = np.array(word_idxs)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           list(np.random.choice(self.count, self.fake_count))

        image_files = self.image_files[current_idxs]
        if self.is_train:
            word_idxs = self.word_idxs[current_idxs]
            masks = self.masks[current_idxs]
            self.current_idx += self.batch_size
            return image_files, word_idxs, masks
        else:
            self.current_idx += self.batch_size
            return image_files

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count

def prepare_train_data(config):
    """ Prepare the data for training the model. """
    coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    if not os.path.exists(config.vocabulary_file):
        vocabulary.build(coco.all_captions())
        vocabulary.save(config.vocabulary_file)
    else:
        vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    coco.filter_by_words(set(vocabulary.words))

    print("Processing the captions...")
    if not os.path.exists(config.temp_annotation_file):
        captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
        image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
        image_files = [os.path.join(config.train_image_dir,
                                    coco.imgs[image_id]['file_name'])
                                    for image_id in image_ids]
        annotations = pd.DataFrame({'image_id': image_ids,
                                    'image_file': image_files,
                                    'caption': captions})
        annotations.to_csv(config.temp_annotation_file)
    else:
        annotations = pd.read_csv(config.temp_annotation_file)
        captions = annotations['caption'].values
        image_ids = annotations['image_id'].values
        image_files = annotations['image_file'].values

    if not os.path.exists(config.temp_data_file):
        word_idxs = []
        masks = []
        for caption in tqdm(captions):
            current_word_idxs_ = vocabulary.process_sentence(caption)
            current_num_words = len(current_word_idxs_)
            current_word_idxs = np.zeros(config.max_caption_length,
                                         dtype = np.int32)
            current_masks = np.zeros(config.max_caption_length)
            current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
            current_masks[:current_num_words] = 1.0
            word_idxs.append(current_word_idxs)
            masks.append(current_masks)
        word_idxs = np.array(word_idxs)
        masks = np.array(masks)
        data = {'word_idxs': word_idxs, 'masks': masks}
        np.save(config.temp_data_file, data)
    else:
        data = np.load(config.temp_data_file).item()
        word_idxs = data['word_idxs']
        masks = data['masks']
    print("Captions processed.")
    print("Number of captions = %d" %(len(captions)))

    print("Building the dataset...")
    dataset = DataSet(image_ids,
                      image_files,
                      config.batch_size,
                      word_idxs,
                      masks,
                      True,
                      True)
    print("Dataset built.")
    return dataset

def prepare_eval_data(config):
    """ Prepare the data for evaluating the model. """
    coco = COCO(config.eval_caption_file)
    image_ids = list(coco.imgs.keys())
    image_files = [os.path.join(config.eval_image_dir,
                                coco.imgs[image_id]['file_name'])
                                for image_id in image_ids]

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size,
                                config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size)
    print("Dataset built.")
    return coco, dataset, vocabulary

def prepare_test_data(config):
    """ Prepare the data for testing the model. """
    files = os.listdir(config.test_image_dir)
    image_files = [os.path.join(config.test_image_dir, f) for f in files
        if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    image_ids = list(range(len(image_files)))

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size,
                                config.vocabulary_file)
    else:
        vocabulary = build_vocabulary(config)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size)
    print("Dataset built.")
    return dataset, vocabulary

def build_vocabulary(config):
    """ Build the vocabulary from the training data and save it to a file. """
    coco = COCO(config.train_caption_file)
    coco.filter_by_cap_len(config.max_caption_length)

    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.build(coco.all_captions())
    vocabulary.save(config.vocabulary_file)
    return vocabulary
