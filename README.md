# CNN-RNN
Tensorflow based implementation of convolution-reccurent network for classification of human interactions on video.
<br>Uses [SDHA 2010 High-level Human Interaction Recognition Challenge](http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html) dataset.

### Requirements

* Python 3.5.2
* Tensorflow > 1.0
* Python OpenCV > 3.0

### Evaluation
Example:
```
python main.py --lrate 0.001 --update
```
Parameter|Default value|Description
---|---|---
`epoch` | 1 | Number of epoch
`esize` | 50 | Size of examples
`estep` | 20 | Length of step for grouping frames into examples
`height` | 240 | Height of frames
`width` | 320 | Width of frames
`lrate` | 1e-4 | Learning rate
`conv` | standard | Type of CNN block (inception/vgg16)
`rnn` | GRU | Type of RNN block (LSTM/GRU)
`update` | False | Re-Generate TFRecords
`download` | False | Download dataset
`restore` | False | Restore from previous checkpoint
`test` | False | Test evaluation
