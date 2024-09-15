# Convolution Neural Network

## Basic Architecture

I'd suggest to read through the official [course notes](https://cs231n.github.io/convolutional-networks/) since it provides really detailed information and
intuitive description.

## Tricks & Notes

**FC -> Conv Layer**

A fully connected layer can be converted into a convolution layer. For example,
consider an output after the final pooling (convolution) process with the size
of [7, 7, 512]. A typical fc layer would flatten the entire matrix and perform
dot product with the next fc layer (typically with size of 4096).

From another point of view, we can apply a 4096, [7, 7] filter with stride 1 on
this output, which yields [1, 1, 4096] output shape. With another 1000, [1, 1]
filter, we can calculate the class scores and give out an output of shape
[1, 1, 1000]. All that we need to do is reshaping it back to 2D.
