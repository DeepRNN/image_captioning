This is an implementation of the neural network architecture for generating image captions as described by the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015). The input is an image, and the output is a sentence describing the content of the image. It first uses a convolutional neural network to extract a feature vector of the input image, and then uses a LSTM recurrent neural network to decode this feature vector into a natural language sentence. A soft attention mechanism is incorporated to improve the quality of the caption. 

This project is implemented in Tensorflow (a previous version uses Caffe to extract visual features and uses Tensorflow to produce words), and allows end-to-end training of both CNN and RNN parts. To use it, you will need the tensorflow version of VGG16 or ResNet(50, 101, 152) model, which can be obtained by using caffe-to-tensorflow (https://github.com/ethereon/caffe-tensorflow). 

References:
1. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.
2. The original implementation in Theano: https://github.com/kelvinxu/arctic-captions
3. Another implementation in Tensorflow: https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow
4. Microsoft COCO dataset: http://mscoco.org/
5. Caffe to Tensorflow: https://github.com/ethereon/caffe-tensorflow

