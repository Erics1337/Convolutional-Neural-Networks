# Convolutional-Neural-Networks
Predicting the classification of Rock, Paper, or Scissors bitmaps using Python Keras Tensorflow

For this project I predicted a series of 32x32 images from a Convolutional Neural Network model that I had created from a training dataset.  

I used k-fold cross validation to split the training dataset and test the trained model on the other part, 
where I was able to achieve over 90% predictive accuracy.

The CNN model consisted of an input layer, a convolutional layer, a max pooling layer, a dropout layer, a hidden layer and an output layer.

I tweaked a few parameters to optimize the effectiveness of the model, 
including adding more nodes to the hidden layer and increasing the number of features generated in the convolutional layer.
