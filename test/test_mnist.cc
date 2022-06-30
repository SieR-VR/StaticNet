/* Copyright 2021- SieR-VR */

#include <iostream>
#include <time.h>

#include "Layers.h"
#include "Datasets.h"

int main(int argc, char *argv[])
{   
    using namespace SingleNet;

    auto MNIST_Image = Image<200, 784>("./Datasets/MNIST/train-images-idx3-ubyte");
    auto MNIST_Label = Label<200>("./Datasets/MNIST/train-labels-idx1-ubyte");

    for (int i = 0; i < 1000; i++)
    {
    }

    auto testImage = Image<200, 784>("./Datasets/MNIST/t10k-images-idx3-ubyte");
    auto testLabel = Label<200>("./Datasets/MNIST/t10k-labels-idx1-ubyte");
}
