This neural system for image captioning is roughly based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015). It is implemented using the Tensorflow library, and allows end-to-end training of both CNN and RNN parts. To use it, you will need the Tensorflow version of VGG16 or ResNet 50/101/152, which can be obtained with Caffe-to-Tensorflow. 

**The code is now compatible with Tensorflow r1.4**.

References
----------

* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.
* [The original implementation in Theano](https://github.com/kelvinxu/arctic-captions)
* [An earlier implementation in Tensorflow](https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow)
* [Microsoft COCO dataset](http://mscoco.org/)
* [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow)

