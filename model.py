import math
import os
import tensorflow as tf
import numpy as np

from base_model import *
from utils.nn import *

class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        with tf.variable_scope("CNN"):
            self.build_cnn()

        with tf.variable_scope("RNN"):
            self.build_rnn()

        if self.mode=="train":
            with tf.variable_scope("Summary"):
                self.build_summary()
        
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
        use_batch_norm = self.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_1_feats = convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = nonlinear(conv1_1_feats, 'relu')
        conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = nonlinear(conv1_2_feats, 'relu')
        pool1_feats = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = nonlinear(conv2_1_feats, 'relu')
        conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = nonlinear(conv2_2_feats, 'relu')
        pool2_feats = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = nonlinear(conv3_1_feats, 'relu')
        conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = nonlinear(conv3_2_feats, 'relu')
        conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = nonlinear(conv3_3_feats, 'relu')
        pool3_feats = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = nonlinear(conv4_1_feats, 'relu')
        conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = nonlinear(conv4_2_feats, 'relu')
        conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = nonlinear(conv4_3_feats, 'relu')
        pool4_feats = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = nonlinear(conv5_1_feats, 'relu')
        conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = nonlinear(conv5_2_feats,  'relu')
        conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = nonlinear(conv5_3_feats, 'relu')

        conv5_3_feats_flat = tf.reshape(conv5_3_feats, [self.batch_size, 196, 512])
        self.conv_feats = conv5_3_feats_flat
        self.conv_feat_shape = [196, 512]
        self.num_ctx = 196                   
        self.dim_ctx = 512

        self.imgs = imgs
        self.is_train = is_train

    def basic_block(self, input_feats, name1, name2, is_train, use_batch_norm, c, s=2):
        """ A basic block of ResNets. """
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, use_batch_norm)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, use_batch_norm)
        branch2a_feats = nonlinear(branch2a_feats, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, use_batch_norm)
        branch2b_feats = nonlinear(branch2b_feats, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, use_batch_norm)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, use_batch_norm, c):
        """ Another basic block of ResNets. """
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, use_batch_norm)
        branch2a_feats = nonlinear(branch2a_feats, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, use_batch_norm)
        branch2b_feats = nonlinear(branch2b_feats, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, use_batch_norm)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_resnet50(self):
        """ Build the ResNet50 net. """
        use_batch_norm = self.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, use_batch_norm)
        conv1_feats = nonlinear(conv1_feats, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, use_batch_norm, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, use_batch_norm, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, use_batch_norm, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, use_batch_norm, 128)
        res3b_feats = self.basic_block2(res3a_feats, 'res3b', 'bn3b', is_train, use_batch_norm, 128)
        res3c_feats = self.basic_block2(res3b_feats, 'res3c', 'bn3c', is_train, use_batch_norm, 128)
        res3d_feats = self.basic_block2(res3c_feats, 'res3d', 'bn3d', is_train, use_batch_norm, 128)

        res4a_feats = self.basic_block(res3d_feats, 'res4a', 'bn4a', is_train, use_batch_norm, 256)
        res4b_feats = self.basic_block2(res4a_feats, 'res4b', 'bn4b', is_train, use_batch_norm, 256)
        res4c_feats = self.basic_block2(res4b_feats, 'res4c', 'bn4c', is_train, use_batch_norm, 256)
        res4d_feats = self.basic_block2(res4c_feats, 'res4d', 'bn4d', is_train, use_batch_norm, 256)
        res4e_feats = self.basic_block2(res4d_feats, 'res4e', 'bn4e', is_train, use_batch_norm, 256)
        res4f_feats = self.basic_block2(res4e_feats, 'res4f', 'bn4f', is_train, use_batch_norm, 256)

        res5a_feats = self.basic_block(res4f_feats, 'res5a', 'bn5a', is_train, use_batch_norm, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, use_batch_norm, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, use_batch_norm, 512)

        res5c_feats_flat = tf.reshape(res5c_feats, [self.batch_size, 49, 2048])
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]
        self.num_ctx = 49                   
        self.dim_ctx = 2048

        self.imgs = imgs
        self.is_train = is_train

    def build_resnet101(self):
        """ Build the ResNet101 net. """
        use_batch_norm = self.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, use_batch_norm)
        conv1_feats = nonlinear(conv1_feats, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, use_batch_norm, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, use_batch_norm, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, use_batch_norm, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, use_batch_norm, 128)       
        temp = res3a_feats
        for i in range(1, 4):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, use_batch_norm, 128)
        res3b3_feats = temp
 
        res4a_feats = self.basic_block(res3b3_feats, 'res4a', 'bn4a', is_train, use_batch_norm, 256)
        temp = res4a_feats
        for i in range(1, 23):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, use_batch_norm, 256)
        res4b22_feats = temp

        res5a_feats = self.basic_block(res4b22_feats, 'res5a', 'bn5a', is_train, use_batch_norm, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, use_batch_norm, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, use_batch_norm, 512)

        res5c_feats_flat = tf.reshape(res5c_feats, [self.batch_size, 49, 2048])
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]
        self.num_ctx = 49                   
        self.dim_ctx = 2048

        self.imgs = imgs
        self.is_train = is_train

    def build_resnet152(self):
        """ Build the ResNet152 net. """
        use_batch_norm = self.use_batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, use_batch_norm)
        conv1_feats = nonlinear(conv1_feats, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, use_batch_norm, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, use_batch_norm, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, use_batch_norm, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, use_batch_norm, 128)       
        temp = res3a_feats
        for i in range(1, 8):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, use_batch_norm, 128)
        res3b7_feats = temp
 
        res4a_feats = self.basic_block(res3b7_feats, 'res4a', 'bn4a', is_train, use_batch_norm, 256)
        temp = res4a_feats
        for i in range(1, 36):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, use_batch_norm, 256)
        res4b35_feats = temp

        res5a_feats = self.basic_block(res4b35_feats, 'res5a', 'bn5a', is_train, use_batch_norm, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, use_batch_norm, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, use_batch_norm, 512)

        res5c_feats_flat = tf.reshape(res5c_feats, [self.batch_size, 49, 2048])
        self.conv_feats = res5c_feats_flat
        self.conv_feat_shape = [49, 2048]
        self.num_ctx = 49                   
        self.dim_ctx = 2048

        self.imgs = imgs
        self.is_train = is_train

    def build_rnn(self):
        """ Build the RNN. """
        if self.mode=="train" or self.beam_size==1:
            self.build_rnn_greedy()
        else:
            self.build_rnn_beam_search()        

    def build_rnn_greedy(self):
        """ Build the RNN using the greedy strategy. """
        print("Building the RNN part...")
        params = self.params

        contexts = self.conv_feats

        sentences = tf.placeholder(tf.int32, [self.batch_size, self.max_sent_len])
        masks = tf.placeholder(tf.float32, [self.batch_size, self.max_sent_len])        
        weights = tf.placeholder(tf.float32, [self.batch_size, self.max_sent_len])        

        # initialize the word embedding
        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] 
                           for i in range(self.num_words)])
        emb_w = weight('emb_weights', [self.num_words, self.dim_embed], init_val=idx2vec)

        # initialize the decoding layer
        dec_w = weight('dec_weights', [self.dim_dec, self.num_words])  
        if self.init_dec_bias: 
            dec_b = bias('dec_biases', [self.num_words], init_val=self.word_table.word_freq)
        else:
            dec_b = bias('dec_biases', [self.num_words], init_val=0.0)
 
        # compute the mean context
        context_mean = tf.reduce_mean(contexts, 1)
       
        # initialize the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(self.dim_hidden, initializer=tf.random_normal_initializer(stddev=0.033)) 
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, self.lstm_keep_prob, self.lstm_keep_prob, self.lstm_keep_prob)

        memory, output = self.init_lstm(context_mean)
        state = memory, output

        cross_entropy_loss = 0.0
        results = []
        scores = []

        alphas = []        
        cross_entropies = []
        num_correct_words = 0.0

        # Generate the words one by one 
        for idx in range(self.max_sent_len):

            # Attention mechanism
            alpha = self.attend(contexts, output)     
                                                                
            masked_alpha = alpha * tf.tile(tf.expand_dims(masks[:, idx], 1), [1, self.num_ctx])        
            alphas.append(tf.reshape(masked_alpha, [-1])) 

            if idx == 0:   
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
                weighted_context = tf.identity(context_mean)
            else:
                word_emb = tf.cond(self.is_train, 
                                   lambda: tf.nn.embedding_lookup(emb_w, sentences[:, idx-1]), 
                                   lambda: word_emb)
                weighted_context = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), 1)
            
            # Apply the LSTM
            with tf.variable_scope("LSTM"):
                output, state = lstm(tf.concat([weighted_context, word_emb], 1), state)
            
            # Compute the logits
            expanded_output = tf.concat([output, weighted_context, word_emb], 1)

            logits1 = fully_connected(expanded_output, self.dim_dec, 'dec_fc')
            logits1 = nonlinear(logits1, 'tanh')
            logits1 = dropout(logits1, self.fc_keep_prob, self.is_train)

            logits2 = tf.nn.xw_plus_b(logits1, dec_w, dec_b)

            # Update the loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sentences[:, idx], 
                                                                           logits=logits2)
            masked_cross_entropy = cross_entropy * masks[:, idx]
            cross_entropy_loss += tf.reduce_sum(masked_cross_entropy*weights[:, idx])
            cross_entropies.append(masked_cross_entropy)

            # Update the result
            max_prob_word = tf.argmax(logits2, 1)
            results.append(max_prob_word)

            is_word_correct = tf.where(tf.equal(max_prob_word, tf.cast(sentences[:, idx], tf.int64)), 
                                       tf.cast(masks[:, idx], tf.float32), 
                                       tf.cast(tf.zeros_like(max_prob_word), tf.float32))
            num_correct_words += tf.reduce_sum(is_word_correct)  

            probs = tf.nn.softmax(logits2) 
            score = tf.log(tf.reduce_max(probs, 1)) 
            scores.append(score) 
            
            # Prepare for the next iteration
            word_emb = tf.cond(self.is_train, lambda: word_emb, lambda: tf.nn.embedding_lookup(emb_w, max_prob_word))          
            tf.get_variable_scope().reuse_variables()                           

        # Get the final result
        results = tf.stack(results, axis=1)
        scores = tf.stack(scores, axis=1)

        alphas = tf.stack(alphas, axis=1)
        alphas = tf.reshape(alphas, [self.batch_size, self.num_ctx, -1])
        sum_alpha = tf.reduce_sum(alphas, axis=2)

        cross_entropies = tf.stack(cross_entropies, axis=1) 
        num_correct_words = num_correct_words / tf.reduce_sum(masks)

        # Compute the final loss 
        cross_entropy_loss = cross_entropy_loss / tf.reduce_sum(masks*weights)

        avg_alpha = tf.reduce_sum(masks, axis=1) / self.num_ctx
        small_alpha_diff = tf.nn.relu(tf.tile(tf.expand_dims(avg_alpha*0.6, 1), [1, self.num_ctx])-sum_alpha)
        large_alpha_diff = tf.nn.relu(sum_alpha-tf.tile(tf.expand_dims(avg_alpha*6, 1), [1, self.num_ctx]))
        attention_loss = tf.nn.l2_loss(small_alpha_diff) + tf.nn.l2_loss(large_alpha_diff) 
        attention_loss = params.att_coeff * attention_loss / self.batch_size 

        if self.train_cnn:
            g_vars = tf.trainable_variables()
        else:
            g_vars = [tf_var for tf_var in tf.trainable_variables() if "CNN" not in tf_var.name]

        l2_loss = params.weight_decay * sum(tf.nn.l2_loss(tf_var) for tf_var in g_vars 
                                                                  if ("bias" not in tf_var.name and
                                                                      "offset" not in tf_var.name and 
                                                                      "scale" not in tf_var.name)) 

        loss = cross_entropy_loss + attention_loss + l2_loss

        # Build the solver 
        with tf.variable_scope("Solver", reuse=tf.AUTO_REUSE):
            learning_rate = tf.train.exponential_decay(params.learning_rate, 
                                                   self.global_step,
                                                   10000, 
                                                   0.9, 
                                                   staircase=True)

            if params.solver=="momentum":
                solver = tf.train.MomentumOptimizer(learning_rate, params.momentum)
            elif params.solver=="rmsprop":
                solver = tf.train.RMSPropOptimizer(learning_rate, params.decay, params.momentum)
            else:
                solver = tf.train.GradientDescentOptimizer(learning_rate)

            gs = tf.gradients(loss, g_vars)
            gs, _ = tf.clip_by_global_norm(gs, 10.0)
            opt_op = solver.apply_gradients(zip(gs, g_vars), global_step=self.global_step)

        self.sentences = sentences
        self.masks = masks
        self.weights = weights

        self.results = results
        self.scores = scores
        self.alphas = alphas

        self.sum_alpha = sum_alpha
        self.cross_entropies = cross_entropies
        self.num_correct_words = num_correct_words

        self.loss = loss
        self.cross_entropy_loss = cross_entropy_loss
        self.attention_loss = attention_loss
        self.l2_loss = l2_loss

        self.opt_op = opt_op
        self.g_vars = g_vars
        self.gs = gs
        
        print("RNN part built.")

    def init_lstm(self, context_mean):
        """Initialize the LSTM using the mean context"""
        temp = context_mean
        for i in range(self.num_init_layers):
            temp = fully_connected(temp, self.dim_hidden, 'init_lstm_fc1'+str(i))
            temp = batch_norm(temp, 'init_lstm_bn1'+str(i), self.is_train, self.use_batch_norm)
            temp = nonlinear(temp, 'tanh')
        memory = tf.identity(temp)
 
        temp = context_mean
        for i in range(self.num_init_layers):
            temp = fully_connected(temp, self.dim_hidden, 'init_lstm_fc2'+str(i))
            temp = batch_norm(temp, 'init_lstm_bn2'+str(i), self.is_train, self.use_batch_norm)
            temp = nonlinear(temp, 'tanh')
        output = tf.identity(temp)

        return memory, output

    def attend(self, contexts, output):
        """Attention Mechanism"""
        context_flat = tf.reshape(contexts, [-1, self.dim_ctx]) 

        context_encode1 = fully_connected(context_flat, self.dim_ctx, 'att_fc11') 
        context_encode1 = batch_norm(context_encode1, 'att_bn11', self.is_train, self.use_batch_norm) 

        context_encode2 = fully_connected_no_bias(output, self.dim_ctx, 'att_fc12') 
        context_encode2 = batch_norm(context_encode2, 'att_bn12', self.is_train, self.use_batch_norm) 
        context_encode2 = tf.tile(tf.expand_dims(context_encode2, 1), [1, self.num_ctx, 1])                 
        context_encode2 = tf.reshape(context_encode2, [-1, self.dim_ctx])    

        context_encode = context_encode1 + context_encode2  
        context_encode = nonlinear(context_encode, 'tanh')  
        context_encode = dropout(context_encode, self.fc_keep_prob, self.is_train)

        alpha = fully_connected_no_bias(context_encode, 1, 'att_fc2')                 
        alpha = batch_norm(alpha, 'att_bn2', self.is_train, self.use_batch_norm)
        alpha = tf.reshape(alpha, [-1, self.num_ctx])                                                           
        alpha = tf.nn.softmax(alpha)

        return alpha

    def build_rnn_beam_search(self):
        """Build the RNN using beam search"""
        print("Building the RNN part...")

        contexts = self.conv_feats

        # initialize the word embedding
        idx2vec = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] 
                           for i in range(self.num_words)])
        emb_w = weight('emb_weights', [self.num_words, self.dim_embed], init_val=idx2vec)

        # initialize the decoding layer
        dec_w = weight('dec_weights', [self.dim_dec, self.num_words])
        if self.init_dec_bias: 
            dec_b = bias('dec_biases', [self.num_words], init_val=self.word_table.word_freq)
        else:
            dec_b = bias('dec_biases', [self.num_words], init_val=0.0)
 
        # compute the mean context
        context_mean = tf.reduce_mean(contexts, 1)
       
        # initialize the LSTM
        memory, output = self.init_lstm(context_mean)

        self.emb_w = emb_w
        self.dec_w = dec_w
        self.dec_b = dec_b
        self.initial_memory = memory
        self.initial_output = output

        # run the RNN for a single step
        self.run_single_step()

    def run_single_step(self):
        """Run the RNN for a single step""" 
        contexts = tf.placeholder(tf.float32, [self.batch_size, self.num_ctx, self.dim_ctx]) 
        last_memory = tf.placeholder(tf.float32, [self.batch_size, self.dim_hidden])
        last_output = tf.placeholder(tf.float32, [self.batch_size, self.dim_hidden])
        last_word = tf.placeholder(tf.int32, [self.batch_size])
        initial_step = tf.placeholder(tf.bool)

        context_mean = tf.reduce_mean(contexts, 1) 

        lstm = tf.nn.rnn_cell.LSTMCell(self.dim_hidden, initializer=tf.random_normal_initializer(stddev=0.033)) 

        # Attention mechanism
        alpha = self.attend(contexts, last_output)                                                                      
        weighted_context = tf.cond(initial_step,
                                   lambda: tf.identity(context_mean),
                                   lambda: tf.reduce_sum(contexts*tf.expand_dims(alpha, 2), 1))

        word_emb = tf.cond(initial_step, 
                           lambda: tf.zeros([self.batch_size, self.dim_embed]), 
                           lambda: tf.nn.embedding_lookup(self.emb_w, last_word))
            
        # Apply the LSTM
        with tf.variable_scope("LSTM"):
            last_state = last_memory, last_output
            output, state = lstm(tf.concat([weighted_context, word_emb], 1), last_state)
            memory, _ = state
            
        # Compute the logits and probs
        expanded_output = tf.concat([output, weighted_context, word_emb], 1)

        logits1 = fully_connected(expanded_output, self.dim_dec, 'dec_fc')
        logits1 = nonlinear(logits1, 'tanh')
        logits2 = tf.nn.xw_plus_b(logits1, self.dec_w, self.dec_b)
        probs = tf.nn.softmax(logits2) 
        logprobs = tf.log(probs)

        tf.get_variable_scope().reuse_variables()                           

        self.contexts = contexts
        self.last_memory = last_memory
        self.last_output = last_output
        self.last_word = last_word
        self.initial_step = initial_step

        self.memory = memory
        self.output = output
        self.logprobs = logprobs
 
    def build_summary(self):
        """Build the summary (for TensorBoard visualization)"""
        assert self.mode=="train"

        for var in tf.trainable_variables():
            with tf.name_scope(var.name[:var.name.find(":")]):
                with tf.name_scope("values"):
                    self.variable_summary(var)

        for g, var in zip(self.gs, self.g_vars):
            with tf.name_scope(var.name[:var.name.find(":")]):
                with tf.name_scope("gradients"):
                    self.variable_summary(g)

        with tf.name_scope("cross_entropies"):
            self.variable_summary(self.cross_entropies)

        with tf.name_scope("attention"):
            self.variable_summary(self.sum_alpha) 

        with tf.name_scope("scores"):
            self.variable_summary(self.scores) 

        tf.summary.scalar("num_correct_words", self.num_correct_words)

        tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
        tf.summary.scalar("attention_loss", self.attention_loss)
        tf.summary.scalar("l2_loss", self.l2_loss)
        tf.summary.scalar("loss", self.loss)
      
        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def get_feed_dict(self, batch, is_train):
        """ Get the feed dictionary for the current batch. """
        if is_train:
            # training phase
            img_files, sentences, masks = batch
            imgs = self.img_loader.load_imgs(img_files)

            weights = []
            for i in range(self.batch_size):
                weights.append(self.word_weight[sentences[i, :]])                
            weights = np.array(weights, np.float32)   

            return {self.imgs: imgs, 
                    self.sentences: sentences, 
                    self.masks: masks, 
                    self.weights: weights, 
                    self.is_train: is_train}

        else:
            # testing or validation phase
            img_files = batch 
            imgs = self.img_loader.load_imgs(img_files)
            fake_sentences = np.zeros((self.batch_size, self.max_sent_len), np.int32)

            return {self.imgs: imgs, 
                    self.sentences: fake_sentences, 
                    self.is_train: is_train}

