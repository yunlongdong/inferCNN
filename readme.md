## infCNN
A pure numpy-based inference framework for CNN. The infCNN supports the inference of model trained on pytorch.

In general, all the elements used in the inference of CNN are divided into ```op``` and ```layer```. ```op``` contains no trainable weights such as "relu, sigmoid, softmax, maxpool, flatten", ```layer``` contains trainable weights such as "conv2d, dense". The ```op``` and ```layer``` implemented are very few now.

#### Inference

Example is shown in ```net.py```, which shows the inference of a LeNet CNN. Weights converted from pytorch model are loaded here. To be noted that ```net.py``` should align with ```train/lenet.py``` since they should have the same model. Just replacing ```nn.Conv2d/Linear``` with ```inferCNN.Conv2d/Dense```   will work.

#### Training on pytorch

The training on pytorch is regular, which can be seen in ```train/``` folder. The weights of CNN are exported to ```.mat``` file.


#### Plugins for ImagePy
The plugin is in ```plgs/``` folder

#### References

* https://github.com/wiseodd/hipsternet
* https://github.com/pytorch/examples/blob/master/mnist/main.py






