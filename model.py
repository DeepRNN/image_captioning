import tensorflow as tf
import numpy as np

from base_model import BaseModel

class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        if self.config.cnn == 'vgg16':
            self.build_vgg16()
        else:
            self.build_resnet50()
        print("CNN built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_1_feats = self.nn.conv2d(images, 64, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                            [config.batch_size, 196, 512])

        self.conv_feats = reshaped_conv5_3_feats
        self.num_ctx = 196
        self.dim_ctx = 512
        self.images = images

    def build_resnet50(self):
        """ Build the ResNet50. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_feats = self.nn.conv2d(images,
                                  filters = 64,
                                  kernel_size = (7, 7),
                                  strides = (2, 2),
                                  activation = None,
                                  name = 'conv1')
        conv1_feats = self.nn.batch_norm(conv1_feats, 'bn_conv1')
        conv1_feats = tf.nn.relu(conv1_feats)
        pool1_feats = self.nn.max_pool2d(conv1_feats,
                                      pool_size = (3, 3),
                                      strides = (2, 2),
                                      name = 'pool1')

        res2a_feats = self.resnet_block(pool1_feats, 'res2a', 'bn2a', 64, 1)
        res2b_feats = self.resnet_block2(res2a_feats, 'res2b', 'bn2b', 64)
        res2c_feats = self.resnet_block2(res2b_feats, 'res2c', 'bn2c', 64)

        res3a_feats = self.resnet_block(res2c_feats, 'res3a', 'bn3a', 128)
        res3b_feats = self.resnet_block2(res3a_feats, 'res3b', 'bn3b', 128)
        res3c_feats = self.resnet_block2(res3b_feats, 'res3c', 'bn3c', 128)
        res3d_feats = self.resnet_block2(res3c_feats, 'res3d', 'bn3d', 128)

        res4a_feats = self.resnet_block(res3d_feats, 'res4a', 'bn4a', 256)
        res4b_feats = self.resnet_block2(res4a_feats, 'res4b', 'bn4b', 256)
        res4c_feats = self.resnet_block2(res4b_feats, 'res4c', 'bn4c', 256)
        res4d_feats = self.resnet_block2(res4c_feats, 'res4d', 'bn4d', 256)
        res4e_feats = self.resnet_block2(res4d_feats, 'res4e', 'bn4e', 256)
        res4f_feats = self.resnet_block2(res4e_feats, 'res4f', 'bn4f', 256)

        res5a_feats = self.resnet_block(res4f_feats, 'res5a', 'bn5a', 512)
        res5b_feats = self.resnet_block2(res5a_feats, 'res5b', 'bn5b', 512)
        res5c_feats = self.resnet_block2(res5b_feats, 'res5c', 'bn5c', 512)

        reshaped_res5c_feats = tf.reshape(res5c_feats,
                                         [config.batch_size, 49, 2048])

        self.conv_feats = reshaped_res5c_feats
        self.num_ctx = 49
        self.dim_ctx = 2048
        self.images = images

    def resnet_block(self, inputs, name1, name2, c, s=2):
        """ A basic block of ResNet. """
        branch1_feats = self.nn.conv2d(inputs,
                                    filters = 4*c,
                                    kernel_size = (1, 1),
                                    strides = (s, s),
                                    activation = None,
                                    use_bias = False,
                                    name = name1+'_branch1')
        branch1_feats = self.nn.batch_norm(branch1_feats, name2+'_branch1')

        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (s, s),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = branch1_feats + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def resnet_block2(self, inputs, name1, name2, c):
        """ Another basic block of ResNet. """
        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = inputs + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            contexts = self.conv_feats
            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])
        else:
            contexts = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, self.num_ctx, self.dim_ctx])
            last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(self.conv_feats, axis = 1)
            initial_memory, initial_output = self.initialize(context_mean)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            num_steps = config.max_caption_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = last_memory, last_output

        # Generate the words one by one
        for idx in range(num_steps):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(contexts, last_output)
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)
                if self.is_train:
                    tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                         [1, self.num_ctx])
                    masked_alpha = alpha * tiled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            # Embed the last word
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([context, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             word_embed],
                                             axis = 1)
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            # Compute the loss for this step, if necessary
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = output
                last_memory = memory
                last_state = state
                last_word = sentences[:, idx]

            tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            alphas = tf.stack(alphas, axis = 1)
            alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis = 2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (config.batch_size * self.num_ctx)

            reg_loss = tf.losses.get_regularization_loss()

            total_loss = cross_entropy_loss + attention_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis = 1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

        self.contexts = contexts
        if self.is_train:
            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            self.attentions = attentions
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs

        print("RNN built.")

    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)
        if config.num_initalize_layers == 1:
            # use 1 fc layer to initialize
            memory = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a')
            output = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b')
        else:
            # use 2 fc layers to initialize
            temp1 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_a1')
            temp1 = self.nn.dropout(temp1)
            memory = self.nn.dense(temp1,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a2')

            temp2 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_b1')
            temp2 = self.nn.dropout(temp2)
            output = self.nn.dense(temp2,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b2')
        return memory, output

    def attend(self, contexts, output):
        """ Attention Mechanism. """
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        if config.num_attend_layers == 1:
            # use 1 fc layer to attend
            logits1 = self.nn.dense(reshaped_contexts,
                                    units = 1,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_a')
            logits1 = tf.reshape(logits1, [-1, self.num_ctx])
            logits2 = self.nn.dense(output,
                                    units = self.num_ctx,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_b')
            logits = logits1 + logits2
        else:
            # use 2 fc layers to attend
            temp1 = self.nn.dense(reshaped_contexts,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1a')
            temp2 = self.nn.dense(output,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1b')
            temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
            temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
            temp = temp1 + temp2
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = 1,
                                   activation = None,
                                   use_bias = False,
                                   name = 'fc_2')
            logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
