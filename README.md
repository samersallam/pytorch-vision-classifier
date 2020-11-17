

# pytorch-vision-classifier	

The main target of this template is to help you during building your classification model where:

1. evaluation_metrics module responsible for calculating and plotting evaluation metrics
2. lr_finder module responsible for finding the best learning rate using the algorithm published by Leslie N. Smith in the paper Cyclical Learning Rates for Training Neural Networks. [The original code](https://github.com/davidtvs/pytorch-lr-finder)
3. pytorch_data_transformation contain the following custom transformations (NRandomCrop, NCenterCrop, LightnessCompensation, Histogram_equalization, CLAHE)
4. pytorch_dataset_preparation module responsible for handling all dataset related functionality and dataset details, whether the dataset in one directory or multiple directories
5. pytorch_dataset_samplers module responsible for samples elements for imbalanced dataset
6. pytorch_device_manager module responsible for handling and viewing available GPU devices details
7. pytorch_loss_function contain the following custom loss functions (EMD)
8. pytorch_model_training module responsible for handling all details related to the training process: 
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