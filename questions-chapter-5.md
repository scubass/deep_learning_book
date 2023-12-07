# Questions

## 1) Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?

The first step, the resize, creates images large enough that they have spare margin to allow further augmentation transforms on their inner regions without creating empty zones. This transformation works by resizing to a square, using a large crop size. On the training set, the crop area is chosen randomly, and the size of the crop is selected to cover the entire width or height of the image, whichever is smaller.

In the second step, the GPU is used for all data augmentation, and all of the potentially destructive operations are done together, with a single interpolation at the end.

## 2) If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.

## 3) What are the two ways in which data is most commonly provided, for most deep learning datasets?

- Individual files representing items of data, such as text documents or images, possibly organized into folders or with filenames representing information about those items
- A table of data, such as in CSV format, where each row is an item which may include filenames providing a connection between the data in the table and data in other formats, such as text documents and images

## 4) Look up the documentation for L and try using a few of the new methods that it adds.

L is a fastai provided extension to the list type in python

```python
p = L.range(20).shuffle()
p
(#20) [5,1,9,10,18,13,6,17,3,16...]

p[2,4,6]
(#3) [9,18,6]

p.argwhere(ge(15))
(#5) [4,7,9,18,19]
```

## 5) Look up the documentation for the Python pathlib module and try using a few methods of the Path class.

Docs: https://docs.python.org/3/library/pathlib.html

```python
p = Path(".")
[x for x in p.iterdir() if x.is_dir()]
```

## 6) Give two examples of ways that image transformations can degrade the quality of the data.

If transformation are performed after resizing down to the augmented size, various common data augmentation transforms might introduce spurious empty zones, degrade data, or both. For instance, rotating an image by 45 degrees fills corner regions of the new bounds with emptiness, which will not teach the model anything. Many rotation and zooming operations will require interpolating to create pixels. These interpolated pixels are derived from the original image data but are still of lower quality

## 7) What method does fastai provide to view the data in a DataLoaders?

```python
dl = DataLoader(data)

dl.show_batch(nrows = 1, ncols = 3)

```

## 8) What method does fastai provide to help you debug a DataBlock?

The method is *summary*, it will attempt to create a batch from the source you give it, with a lot of details. Also, if it fails, you will see exactly at which point the error happens, and the library will try to give you some help.

```python
pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'))
pets1.summary(path/"images")
```

## 9) Should you hold off on training a model until you have thoroughly cleaned your data?

No, you should train your model as early as possible, because you might find that you don't need a lot work to acomplish your task, and if you happend to do need extra work, you can analyze your trained model to better understand what you need to do.

## 10) What are the two pieces that are combined into cross-entropy loss in PyTorch?

softmax and ned log loss

## 11) What are the two properties of activations that softmax ensures? Why is this important?

Taking the exponential ensures all our numbers are positive, and then dividing by the sum ensures we are going to have a bunch of numbers that add up to 1. The exponential also has a nice property: if one of the numbers in our activations x is slightly bigger than the others, the exponential will amplify this.

### Properties:

- Non-negativity
- Normalization
- Mutual exclusivity
- Sensitivity to input changes
- Sensitivity
- Differentiability

## 12) When might you want your activations to not have these two properties?

## 13) Calculate the exp and softmax columns of <> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).

## 14) Why can't we use torch.where to create a loss function for datasets where our label can have more than two categories?

## 15) What is the value of log(-2)? Why?

## 16) What are two good rules of thumb for picking a learning rate from the learning rate finder?

## 17) What two steps does the fine_tune method do?

## 18) In Jupyter Notebook, how do you get the source code for a method or function?

## 19) What are discriminative learning rates?

## 20) How is a Python slice object interpreted when passed as a learning rate to fastai?

## 21) Why is early stopping a poor choice when using 1cycle training?

## 22) What is the difference between resnet50 and resnet101?

## 23) What does to_fp16 do?
