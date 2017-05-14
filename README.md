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
python main.py --lrate 0.001 --update
```
Optional parameters:<br>
`epoch`, 1, 'Number of epoch' <br>
`esize`, 50, 'Size of examples' <br>
`estep`, 20, 'Length of step for grouping frames into examples'<br>
`height`, 240, 'Height of frames'<br>
`width`, 320, 'Width of frames'<br>
`lrate`, 1e-4, 'Learning rate'<br>
`conv`, 'standard', 'Type of CNN block'<br>
`rnn`, 'GRU', 'Type of RNN block (LSTM/GRU)'<br>
`update`, False, 'Generate TFRecords'<br>
`download`, False, 'Download dataset'<br>
`restore`, False, 'Restore from previous checkpoint'<br>
`test`, False, 'Test evaluation'<br>