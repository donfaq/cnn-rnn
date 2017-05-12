# CNN-RNN
Tensorflow based implementation of convolution-reccurent network for classification of human interactions on video.
<br>Uses [SDHA 2010 High-level Human Interaction Recognition Challenge](http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html) dataset.

### Requirements

* Python 3.5.2
* Tensorflow > 1.0
* Python OpenCV > 3.0

### Run training
Example:
```
python main.py -esize 50 -estep 20 -height 240 -width 320 -lrate 1e-4 -d -u
```
`-esize` Size of examples<br>
`-estep` Size of step for grouping frames into examples<br>
`-height` Height of frames<br>
`-width` Width of frames<br>
`-lrate` Learning rate<br>
`-u`, `--update` Re-create tfrecords<br>
`-d`, `--download`  Download dataset<br>
`-r`, `--restore`,  Re-store checkpoint<br>
`-cnn inception/vgg16`, To customize cnn block
`-is_training`, To train network (nothing for test from last checkpoint)