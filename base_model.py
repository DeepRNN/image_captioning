import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from dataset import *
from utils.words import *
from utils.coco.coco import *
from utils.coco.pycocoevalcap.eval import *

class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True 
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)

    def load_img(self, img_file):    
        """ Load and preprocess an image. """  
        img = cv2.imread(img_file)

        if self.bgr:
            temp = img.swapaxes(0, 2)
            temp = temp[::-1]
            img = temp.swapaxes(0, 2)

        img = cv2.resize(img, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        img = img[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1], :]
        img = img - self.mean
        return img

    def load_imgs(self, img_files):
        """ Load and preprocess a list of images. """
        imgs = []
        for img_file in img_files:
            imgs.append(self.load_img(img_file))
        imgs = np.array(imgs, np.float32)
        return imgs

class BaseModel(object):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.batch_size = params.batch_size

        self.cnn_model = params.cnn_model
        self.train_cnn = params.train_cnn
        self.class_balancing_factor = params.class_balancing_factor

        self.save_dir = params.save_dir

        self.word_table = WordTable(params.vocab_size, 
                                    params.dim_embed, 
                                    params.max_sent_len, 
                                    params.word_table_file)
        self.word_table.load()

        self.img_loader = ImageLoader(params.mean_file)
        self.img_shape = [224, 224, 3]

        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

        self.build()
  
    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def train(self, sess, train_coco, train_data):
        """ Train the model. """
        print("Training the model...")
        params = self.params
        num_epochs = params.num_epochs

        train_writer = tf.summary.FileWriter("./", sess.graph)
        for epoch_no in tqdm(list(range(num_epochs)), desc='epoch'):
            for idx in tqdm(list(range(train_data.num_batches)), desc='batch'):

                batch = train_data.next_batch()
                feed_dict = self.get_feed_dict(batch, is_train=True)
                _, summary, global_step = sess.run([self.opt_op,                                                                     
                                                    self.summary, 
                                                    self.global_step], 
                                                    feed_dict=feed_dict)

                if (global_step + 1) % params.save_period == 0:
                    self.save(sess)
                
                train_writer.add_summary(summary, global_step)

            train_data.reset()

        self.save(sess)
        train_writer.close()

        print("Training complete.")

    def val(self, sess, val_coco, val_data, save_result_as_img=False):
        """ Validate the model. """
        print("Validating the model ...")
        results = []
        result_dir = self.params.val_result_dir

        # Generate the captions for the images
        cur_ind = 0
        for k in tqdm(list(range(val_data.num_batches))):
            batch = val_data.next_batch()
            feed_dict = self.get_feed_dict(batch, is_train=False)
            result = sess.run(self.results, feed_dict=feed_dict)

            fake_cnt = 0 if k<val_data.num_batches-1 else val_data.fake_count
            for l in range(val_data.batch_size-fake_cnt):            
                sentence = self.word_table.indices_to_sent(result[l])
                results.append({'image_id': val_data.img_ids[cur_ind], 'caption': sentence})
                cur_ind += 1 

                # Save the result in an image file
                if save_result_as_img:
                    img_file = batch[l]
                    img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]
                    img = mpimg.imread(img_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(sentence)
                    plt.savefig(os.path.join(result_dir, img_name+'_result.jpg'))

        val_data.reset() 

        # Evaluate these captions
        val_res_coco = val_coco.loadRes2(results)
        scorer = COCOEvalCap(val_coco, val_res_coco)
        scorer.evaluate()
        print("Validation complete.")

    def test(self, sess, test_data, save_result_as_img=True):
        """ Test the model. """
        print("Testing the model ...")
        result_file = self.params.test_result_file
        result_dir = self.params.test_result_dir
        captions = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches))):
            batch = test_data.next_batch()
            feed_dict = self.get_feed_dict(batch, is_train=False)
            result = sess.run(self.results, feed_dict=feed_dict)

            fake_cnt = 0 if k<test_data.num_batches-1 else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):            
                sentence = self.word_table.indices_to_sent(result[l])
                captions.append(sentence)
        
                # Save the result in an image file
                if save_result_as_img:
                    img_file = batch[l]
                    img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]
                    img = mpimg.imread(img_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(sentence)
                    plt.savefig(os.path.join(result_dir, img_name+'_result.jpg'))

        # Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.img_files, 'caption':captions})
        results.to_csv(result_file)
        print("Testing complete.")
 
    def save(self, sess):
        """ Save the model. """
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(self.save_dir, str(self.global_step.eval()))  

        print((" Saving the model to %s ..." % (save_path+".npy")))

        np.save(save_path, data)
        info_path = os.path.join(self.save_dir, "info")  
        info_file = open(info_path, "wb")
        info_file.write(str(self.global_step.eval()))
        info_file.close()          

        print("Model saved.")

    def load(self, sess):
        """ Load the model. """
        if self.params.model_file is not None:
            save_path = self.params.model_file        
        else:
            info_path = os.path.join(self.save_dir, "info")  
            info_file = open(info_path, "rb")
            global_step = info_file.read()
            info_file.close() 
            save_path = os.path.join(self.save_dir, global_step+".npy")  
 
        print("Loading the model from %s ..." % save_path)
   
        data_dict = np.load(save_path).item()
        for v in tf.global_variables(): 
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))

        print("Model loaded.")
   
    def load_cnn(self, data_path, session, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading CNN model from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        with tf.variable_scope("CNN", reuse=True):
            for op_name in data_dict:
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].iteritems():
                        try:
                            var = tf.get_variable(param_name)
                            session.run(var.assign(data))
                            count += 1
                        except ValueError:
                            if not ignore_missing:
                                raise
        print("%d tensors loaded. " %count)

