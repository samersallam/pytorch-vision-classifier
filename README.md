

# pytorch-vision-classifier

![image](https://cdn-images-1.medium.com/max/1000/1*aqNgmfyBIStLrf9k7d9cng.jpeg)

The main target of this template is to help you build your classification model quickly.

in this library, you can find the following modules:

1. lr_finder module is responsible for finding the best learning rate using the algorithm published by Leslie N. Smith in the paper Cyclical Learning Rates for Training Neural Networks. [The original code](https://github.com/davidtvs/pytorch-lr-finder)
2. pytorch_data_transformation module contains the following customized transformations (NRandomCrop, NCenterCrop, LightnessCompensation, Histogram_equalization, CLAHE)
3. pytorch_dataset_preparation module is responsible for handling all dataset related functionalities and dataset details, whether the dataset is in one directory or multiple directories
4. pytorch_dataset_samplers module has all the customized samplers
5. pytorch_device_manager module is responsible for handling and viewing available GPU devices details
6. pytorch_loss_function contains the following customized loss functions (EMD)
7. pytorch_model_training module is responsible for handling all details related to the training process: 
	a. Get a pre-trained model for you with the last layer updated with or without a dropout layer and initialized by an algorithm you choose from the most common initialization algorithms
	b. Know the GPU memory usage of your model
	c. Understand the timing of different steps during model training
	d. Provides a dashboard to monitor your model during the training process
	e. Track the metric that you choose to find the best model for your problem
	f. Provides a compressed version of your model to be used for deployment purpose
	g. Extract features from last layer, classification layer, or softmax layer


In order to install, you need to download [pytorch](https://pytorch.org/get-started/locally/).
then open the command prompt and type:
```
pip install pytorch_vision_classifier
```

Refer to the following [notebook](https://github.com/Samer92/pytorch-vision-classifier/tree/master/example) to see some code examples