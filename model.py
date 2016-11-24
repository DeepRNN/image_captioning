import math
import os
import tensorflow as tf
import numpy as np

from base_model import *
from utils.nn import *

class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN part...")

        if self.cnn_model=='vgg16':
            self.build_vgg16()

        elif self.cnn_model=='resnet50':
            self.build_resnet50()

        elif self.cnn_model=='resnet101':
            self.build_resnet101()

        else:
            self.build_resnet152()

        print("CNN part built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_1_feats = convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
        conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
        conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
        pool2_feats = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
        conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
        conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
        pool3_feats = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
        conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
        conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
        pool4_feats = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
        conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
        conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

        pool5_feats = max_pool(conv5_3_feats, 2, 2, 2, 2, 'pool5')
        pool5_feats_flat = tf.reshape(pool5_feats, [self.batch_size, -1]) 

        pool5_feats_flat.set_shape([self.batch_size, 49*512])
        fc6_feats = fully_connected(pool5_feats_flat, 4096, 'fc6')
        fc6_feats = nonlinear(fc6_feats, 'relu')
        if self.train_cnn:
            fc6_feats = dropout(fc6_feats, 0.5, is_train)

        fc7_feats = fully_connected(fc6_feats, 4096, 'fc7')

        conv5_3_feats_flat = tf.reshape(conv5_3_feats, [self.batch_size, 196, 512])
        self.conv_feats = conv5_3_feats_flat
        self.conv_feat_shape = [196, 512]

        self.fc_feats = fc7_feats
        self.fc_feat_shape = [4096]

        self.imgs = imgs
        self.is_train = is_train

    def basic_block(self, input_feats, name1, name2, is_train, bn, c, s=2):
        """ A basic block of ResNets. """
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, bn, None)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, bn, c):
        """ Another basic block of ResNets. """
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_resnet50(self):
        """ Build the ResNet50 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)
        res3b_feats = self.basic_block2(res3a_feats, 'res3b', 'bn3b', is_train, bn, 128)
        res3c_feats = self.basic_block2(res3b_feats, 'res3c', 'bn3c', is_train, bn, 128)
        res3d_feats = self.basic_block2(res3c_feats, 'res3d', 'bn3d', is_train, bn, 128)

        res4a_feats = self.basic_block(res3d_feats, 'res4a', 'bn4a', is_train, bn, 256)
        res4b_feats = self.basic_block2(res4a_feats, 'res4b', 'bn4b', is_train, bn, 256)
        res4c_feats = self.basic_block2(res4b_feats, 'res4c', 'bn4c', is_train, bn, 256)
        res4d_feats = self.basic_block2(res4c_feats, 'res4d', 'bn4d', is_train, bn, 256)
        res4e_feats = self.basic_block2(res4d_feats, 'res4e', 'bn4e', is_train, bn, 256)
        res4f_feats = self.basic_block2(res4e_feats, 'res4f', 'bn4f', is_train, bn, 256)

        res5a_feats = self.basic_block(res4f_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        res5c_feats_flat = tf.reshape(res5c_feats, [self.batch_size, 49, 2048])
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = imgs
        self.is_train = is_train

    def build_resnet101(self):
        """ Build the ResNet101 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 4):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b3_feats = temp
 
        res4a_feats = self.basic_block(res3b3_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 23):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b22_feats = temp

        res5a_feats = self.basic_block(res4b22_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        res5c_feats_flat = tf.reshape(res5c_feats, [self.batch_size, 49, 2048])
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = imgs
        self.is_train = is_train

    def build_resnet152(self):
        """ Build the ResNet152 net. """
        bn = self.params.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 8):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b7_feats = temp
 
        res4a_feats = self.basic_block(res3b7_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 36):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b35_feats = temp

        res5a_feats = self.basic_block(res4b35_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        res5c_feats_flat = tf.reshape(res5c_feats, [self.batch_size, 49, 2048])
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]

        self.imgs = imgs
        self.is_train = is_train

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN part...")

        params = self.params
        bn = params.batch_norm      

        batch_size = self.batch_size                        
        num_ctx = self.conv_feat_shape[0]                   
        dim_ctx = self.conv_feat_shape[1]                   

        num_words = self.word_table.num_words
        max_sent_len = params.max_sent_len
        num_lstm = params.num_lstm
        dim_embed = params.dim_embed
        dim_hidden = params.dim_hidden
        dim_dec = params.dim_dec

        if not self.train_cnn:
            contexts = tf.placeholder(tf.float32, [batch_size] + self.conv_feat_shape)
            if self.init_lstm_with_fc_feats:
                feats = tf.placeholder(tf.float32, [batch_size] + self.fc_feat_shape)
        else:
            contexts = self.conv_feats
            if self.init_lstm_with_fc_feats:
                feats = self.fc_feats

        sentences = tf.placeholder(tf.int32, [batch_size, max_sent_len])
        masks = tf.placeholder(tf.float32, [batch_size, max_sent_len])        

        is_train = self.is_train

        self.word_weight = np.exp(-np.array(self.word_table.word_freq)*self.class_balancing_factor)

        self.position_weight = np.exp(-np.array(list(range(max_sent_len)))*0.003)

        # initialize the word embedding
        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] for i in range(num_words)])
        if params.fix_embed_weight:
            emb_w = tf.convert_to_tensor(idx2vec, tf.float32)
        else:
            emb_w = weight('emb_w', [num_words, dim_embed], init_val=idx2vec, group_id=1)

        # initialize the decoding layer
        dec_w = weight('dec_w', [dim_dec, num_words], group_id=1)  
        if params.init_dec_bias: 
            dec_b = bias('dec_b', [num_words], init_val=self.word_table.word_freq)
        else:
            dec_b = bias('dec_b', [num_words], init_val=0.0)
 
        # compute the mean context
        context_mean = tf.reduce_mean(contexts, 1)
       
        # initialize the LSTMs
        lstm = tf.nn.rnn_cell.LSTMCell(dim_hidden, initializer=tf.random_normal_initializer(stddev=0.03)) 

        if self.init_lstm_with_fc_feats:
            init_feats = feats
        else:
            init_feats = context_mean

        if num_lstm == 1:
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc1'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn1'+str(i), is_train, bn, 'tanh')
            memory = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc2'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn2'+str(i), is_train, bn, 'tanh')
            output = tf.identity(temp)

            state = tf.nn.rnn_cell.LSTMStateTuple(memory, output)                   

        else:
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc11'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn11'+str(i), is_train, bn, 'tanh')
            memory1 = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc12'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn12'+str(i), is_train, bn, 'tanh')
            output1 = tf.identity(temp)

            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc21'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn21'+str(i), is_train, bn, 'tanh')
            memory2 = tf.identity(temp)
 
            temp = init_feats
            for i in range(params.num_init_layers):
                temp = fully_connected(temp, dim_hidden, 'init_lstm_fc22'+str(i), group_id=1)
                temp = batch_norm(temp, 'init_lstm_bn22'+str(i), is_train, bn, 'tanh')
            output = tf.identity(temp)

            state1 = tf.nn.rnn_cell.LSTMStateTuple(memory1, output1)                
            state2 = tf.nn.rnn_cell.LSTMStateTuple(memory2, output)                 

        loss0 = 0.0
        results = []
        scores = []
        context_flat = tf.reshape(contexts, [-1, dim_ctx])  
       
        # Generate the words one by one 
        for idx in range(max_sent_len):

            # Attention mechanism
            context_encode1 = fully_connected(context_flat, dim_ctx, 'att_fc11', group_id=1) 
            context_encode1 = batch_norm(context_encode1, 'att_bn11', is_train, bn, None) 

            context_encode2 = fully_connected_no_bias(output, dim_ctx, 'att_fc12', group_id=1) 
            context_encode2 = batch_norm(context_encode2, 'att_bn12', is_train, bn, None) 
            context_encode2 = tf.tile(tf.expand_dims(context_encode2, 1), [1, num_ctx, 1])                 
            context_encode2 = tf.reshape(context_encode2, [-1, dim_ctx])    

            context_encode = context_encode1 + context_encode2  
            context_encode = nonlinear(context_encode, 'relu')  
            context_encode = dropout(context_encode, 0.5, is_train)

            alpha = fully_connected(context_encode, 1, 'att_fc2', group_id=1)                 
            alpha = batch_norm(alpha, 'att_bn2', is_train, bn, None)
            alpha = tf.reshape(alpha, [-1, num_ctx])                                                           
            alpha = tf.nn.softmax(alpha)                                                                       
         
            if idx == 0:   
                word_emb = tf.zeros([batch_size, dim_embed])
                weighted_context = tf.identity(context_mean)
            else:
                word_emb = tf.cond(is_train, lambda: tf.nn.embedding_lookup(emb_w, sentences[:, idx-1]), lambda: word_emb)
                weighted_context = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), 1)
            
            # Apply the LSTMs 
            if num_lstm == 1:
                with tf.variable_scope("lstm"):
                    output, state = lstm(tf.concat(1, [weighted_context, word_emb]), state)
            else:
                with tf.variable_scope("lstm1"):
                    output1, state1 = lstm(weighted_context, state1)

                with tf.variable_scope("lstm2"):
                    output, state2 = lstm(tf.concat(1, [word_emb, output1]), state2)
            
            # Compute the logits
            expanded_output = tf.concat(1, [output, weighted_context, word_emb])

            logits1 = fully_connected(expanded_output, dim_dec, 'dec_fc', group_id=1)
            logits1 = nonlinear(logits1, 'tanh')
            logits1 = dropout(logits1, 0.5, is_train)

            logits2 = tf.nn.xw_plus_b(logits1, dec_w, dec_b)

            # Update the loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, sentences[:, idx])
            cross_entropy = cross_entropy * masks[:, idx]
            loss0 += tf.reduce_sum(cross_entropy)
 
            # Update the result
            max_prob_word = tf.argmax(logits2, 1)
            results.append(max_prob_word)

            probs = tf.nn.softmax(logits2)
            score = tf.reduce_max(probs, 1)
            scores.append(score)

            # Prepare for the next iteration
            word_emb = tf.cond(is_train, lambda: word_emb, lambda: tf.nn.embedding_lookup(emb_w, max_prob_word))          
            tf.get_variable_scope().reuse_variables()                           

        # Get the final result
        results = tf.pack(results, axis=1)
        scores = tf.pack(scores, axis=1)

        # Compute the final loss 
        loss0 = loss0 / tf.reduce_sum(masks)
        if self.train_cnn:
            loss1 = params.weight_decay * (tf.add_n(tf.get_collection('l2_0')) + tf.add_n(tf.get_collection('l2_1')))
        else:
            loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_1'))
        loss = loss0 + loss1
        
        # Build the solver
        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        tvars = tf.trainable_variables()
        gs, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 3.0)
        opt_op = solver.apply_gradients(zip(gs, tvars), global_step=self.global_step)

        self.contexts = contexts
        if self.init_lstm_with_fc_feats:
            self.feats = feats
        self.sentences = sentences
        self.masks = masks

        self.loss = loss
        self.loss0 = loss0
        self.loss1 = loss1
        self.opt_op = opt_op

        self.results = results
        self.scores = scores
        
        print("RNN part built.")        

    def get_feed_dict(self, batch, is_train, contexts=None, feats=None):
        """ Get the feed dictionary for the current batch. """
        if is_train:
            # training phase
            img_files, sentences, masks = batch
            imgs = self.img_loader.load_imgs(img_files)

            for i in range(self.batch_size):
                word_weight = self.word_weight[sentences[i, :]]                
                masks[i, :] = masks[i, :] * word_weight
                masks[i, :] = masks[i, :] * self.position_weight

            if self.train_cnn:
                return {self.imgs: imgs, self.sentences: sentences, self.masks: masks, self.is_train: is_train}
            else:
                if self.init_lstm_with_fc_feats:
                    return {self.contexts: contexts, self.feats: feats, self.sentences: sentences, self.masks: masks, self.is_train: is_train}        
                else:
                    return {self.contexts: contexts, self.sentences: sentences, self.masks: masks, self.is_train: is_train} 

        else:
            # testing or validation phase
            img_files = batch 
            imgs = self.img_loader.load_imgs(img_files)
            fake_sentences = np.zeros((self.batch_size, self.params.max_sent_len), np.int32)

            if self.train_cnn:
                return {self.imgs: imgs, self.sentences: fake_sentences, self.is_train: is_train}
            else:
                if self.init_lstm_with_fc_feats:
                    return {self.contexts: contexts, self.feats: feats, self.sentences: fake_sentences, self.is_train: is_train}        
                else:
                    return {self.contexts: contexts, self.sentences: fake_sentences, self.is_train: is_train} 


