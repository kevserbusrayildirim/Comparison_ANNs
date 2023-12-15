# Comparison_ANNs
Experimental Comparisons of Artificial Neural Networks Optimization Algorithms

Multi-Layer Perceptron (MLP) created on the CIFAR-10 dataset was trained with different optimizer algorithms and different learning rate values. The performance of the model was evaluated through graphs and a confusion matrix.

Input Layer:
•	Input Size: 32x32x3 (32 pixels width, 32 pixels height, and 3 color channels RGB)
•	Output Size: 1000 neurons
•	Activation Function: ReLU (Rectified Linear Unit)

Hidden Layer:
•	Input Size: 1000 neurons
•	Output Size: 512 neurons
•	Activation Function: ReLU (Rectified Linear Unit)

Output Layer:
•	Input Size: 512 neurons
•	Output Size: 10 neurons (Number of classes in the dataset)
•	Activation Function: Linear (Typically, SoftMax is not used in the output layer, as cross-entropy loss already incorporates SoftMax)

 Parameter Setting
 
•	Loss function: Cross Entropy Loss
•	Optimization applications: RMSprop, SGD, Adam, Adagrad (With specified learning rates: 1e-2 & 1e-6)
•	Number of periods: 100
•	Batch size: 128
•	L2 Regularization: weight_decay=1e-5

Dataset

CIFAR-10 is a widely used image classification dataset created by the "Canadian Institute for Advanced Research" (CIFAR). This dataset contains a total of 60,000 color (RGB) 32x32 pixel images from 10 different classes. Each class contains 6,000 images. It is a widely used dataset for training and evaluating image classification algorithms.

The classes of CIFAR-10 are:
•	Airplane
•	Automobile
•	Bird
•	Cat
•	Deer
•	Dog
•	Frog
•	Horse
•	Ship
•	Truck


