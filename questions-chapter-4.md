It is very important that when we are working with neural networks all the data should be a float number between 0 and 1, so we usually have to take some steps to normalize the data and avoid negative numbers.

## 1) How is a grayscale image represented on a computer? How about a color image?

It is represneted as a matrix of 8 bits integers, and a color image is represented as a matrix of a 8 bits by 3 tuple of colors, (RED, GREEN, BLUE)

## 2) How are the files and folders in the MNIST_SAMPLE dataset structured? Why?

We have a folder with the trainning data and validation data, we need to separate them because we will train the model on some data and validate it on another.

## 3) Explain how the "pixel similarity" approach to classifying digits works.

We take the average of all the images of a specific digit, thus creating the "ideal" image for a digit, then in order to make a prediction we calculate the distance from all the "ideal" digits and the smallest distance is the prediction.

### Functions to calculate the distance

- Take the mean of the absolute value of differences (absolute value is the function that replaces negative values with positive values). This is called the mean absolute difference or L1 norm
- Take the mean of the square of differences (which makes everything positive) and then take the square root (which undoes the squaring). This is called the root mean squared error (RMSE) or L2 norm.

## 4) What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.

List comprehesion is a feature that python took from haskell, is basically a way to produce a list from an iterator.

### Example

```python
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
[2 * x for x in nums if x % 2 == 0]
```

## 5) What is a "rank-3 tensor"?

A stack of matrices, or a stack of rank-2 tensors

## 6) What is the difference between tensor rank and shape? How do you get the rank from the shape?

### Imagine a bookshelf.

- Rank tells you how many shelves it has (e.g., 3 shelves).
- Shape tells you how many books are on each shelf (e.g., (5, 7, 2), meaning the first shelf has 5 books, the second has 7, and the third has 2).

Rank is the overall structure of the data (number of dimensions), while shape describes the specific size of each dimension. Both are crucial for understanding and manipulating tensors in your code.

## 7) What are RMSE and L1 norm?

Root mean squared error or L2 norm, and L1 norm or mean absoulte difference are functions that data scientist use to calculate the distance

## 8) How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

You can do it on a gpu if because it has a couple thousand cores that do some very simple math many times per second and if you can paralellize an operation then the gpu will be an ideal candidate. In the case of AI using a gpu we can multiplicate matrices in a parallel way very fast. That is why they are used to calculate the wheits mat_mul input plus bias and the gradient for each wheight.

## 9) Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.

```python
import torch

tensor = torch.arange(10, 10) # or list(range(1, 10))

tensor_3x3 = tensor.view(3, 3)

bottom_two_rows = tensor[1:]

without_first_column = bottom_two_rows[:, -2]

# also we could chain it all toghether like this: 
# tensor_3x3[1:][:, -2]
```

## 10) What is broadcasting?

Tensor arguments can automatically be expanded to be of equal size in order to make an operation

### Rules of broadcasting:

- Each tensor has at least one dimension.

- When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.


#### Example 
##### can line up trailing dimensions
>>> x=torch.empty(5,3,4,1)
>>> y=torch.empty(  3,1,1)
##### x and y are broadcastable.
##### 1st trailing dimension: both have size 1
##### 2nd trailing dimension: y has size 1
##### 3rd trailing dimension: x size == y size
##### 4th trailing dimension: y dimension doesn't exist

##### but:
>>> x=torch.empty(5,2,4,1)
>>> y=torch.empty(  3,1,1)
##### x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

## 11) Are metrics generally calculated using the training set, or the validation set? Why?

## 12) What is SGD?

Stochastig gradient descent is the prefered way to optimize a neural net, but becuase a neural network is just a function it capable of optimizing any function

## 13) Why does SGD use mini-batches?

Calculating SGD for every step is very expensive, so what we batch out data and take the average of the loss function and use that to back propagate.

## 14) What are the seven steps in SGD for machine learning?

#hide_input
#id gradient_descent
#caption The gradient descent process
#alt Graph showing the steps for Gradient Descent
gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')

## 15) How do we initialize the weights in a model?

We randomize them, since we already have a method to optimize them, we really don't need to spend much time finding a good place to initialize them, and there are always a number between 0 and 1.

## 16) What is "loss"?

It is a mesure of how good is the model predicting the targets for the inputs. The convention is that the loss evaluates to a number between 0 and 1 and the lower it is the more acurate the model.
The goal of training is to minimize the loss over the entire training data, leading the network towards making better predictions.
It is important to note that loss is not the most intuitive measure for a human to understand. So to show humans how well is the model at a given point we use acurracy, that is the percentage of correct predictions made by the model.

## 17) Why can't we always use a high learning rate?

Because we need to find a local minima of the function, and if we jump by a big number we might miss the local minima and overextend (or overfit idk)

Validation set, because the model can memorize the training set, thus we want to tets our model with data it hasn't seen before.

## 18) What is a "gradient"?

It is a function that tells us the direction of a local minima, it returns a vector that tells you in which direction and by how much you need to change the Xs to increase a specific value. (that is why when we update the parameters we subtract)

## 19) Do you need to know how to calculate gradients yourself?

No pytorch already knows how to calculate the gradient of almost all functions.

## 20) Why can't we use accuracy as a loss function?

### There are several reasons:

-  Not differentiable

- Uninformative for small changes

Basically it serves to show us, the humans how good is the model predictions but in order to use SGD we need another function that plays well with being differentiable and that doesn't have big jumps like the sigmoid function.

## 21) Draw the sigmoid function. What is special about its shape?

It is a very smoth curve and it returns at most 1 an at lead 0.

## 22) What is the difference between a loss function and a metric?

A very small change in the value of a weight will often not actually change the accuracy at all. This means it is not useful to use accuracy as a loss function—if we do, most of the time our gradients will actually be 0, and the model will not be able to learn from that number. 
Instead, we need a loss function which, when our weights result in slightly better predictions, gives us a slightly better loss

## 23) What is the function to calculate new weights using a learning rate?

w -= gradient(w) * lr

## 24) What does the DataLoader class do?

It takes any python collection and turns it into an iterator over mini-batches

### Example

```python
collection = range(10)
dl = DataLoader(collection, batch_size=5)
list(dl)
```

## 26) Write pseudocode showing the basic steps taken in each epoch for SGD.

### Steps:

1) calculate the forward step, that is w @ xs + bias for each parameter
2) calculate loss function
3) call loss.backward() to calculate the gradient of each parameter
4) iterate over all the parameters (wheights) and update them
5) don't forget to set the gradient to zero

## 27) Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?

```python
def zip_collections(l, r):
    return zip(l, r)

assert zip_collections([1, 2, 3, 4], 'abcd') == [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
```

## 28) What are the "bias" parameters in a neural network? Why do we need them?

Biases are constans values added to to wheights * inputs before applying the activation function in each neuron.

### Why do we need biases:

In some cases, the weights connecting neurons might become symmetrical, leading to situations where the neuron always outputs the same value regardless of the input.
Biases break this symmetry by adding a constant offset, allowing the neuron to differentiate between different input patterns and learn non-trivial relationships.
By providing a starting point for the activation function, biases reduce the amount of adjustment needed in the weights to reach the desired output.

## 29) What does the @ operator do in Python?

Matrix multiplication

## 30) What does the backward method do?

It calcualtes the gradient of the loss functions with respect of the parameters of the neural network.

## 31) Why do we have to zero the gradients?

Because when calculating the gradients of each parameter we add them, so if we don't zero them we will be working with the addition of the last gradients and the current ones

## 32) What information do we have to pass to Learner?

To create a Learner without using an application (such as vision_learner) we need to pass in all the elements that we've created in this chapter: the *DataLoaders*, the *model*, the *optimization function* (which will be passed the parameters), the *loss function*, and optionally any *metrics* to print:

## 33) Show Python or pseudocode for the basic steps of a training loop.

1) dot product of the inputs and parameters
2) calculate loss function
3) backwards pass (modify the parameters according to the gradients and learning rate)
4) zero the gradients for each parameter

## 34) What is "ReLU"? Draw a plot of it for values from -2 to +2. (rectified linear unit (ReLU))

A function that outputs the input if the input is positive and 0 otherwise.

## 35) What is an "activation function"?

It is a function that determines if the function should fire or not.

### Common activation functions:

- ReLU (Rectified Linear Unit)
- Sigmoid
- Tanh
- Softmax

## 36) What's the difference between F.relu and nn.ReLU?

They are the same, just that F.relu is a function and nn.ReLU is a module, in pytorch a lot of functions have their own modules too

## 37) The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?

Because when we add nonlinearity and more layers it it proven that the model can learn more complex patterns better, and usually faster.
