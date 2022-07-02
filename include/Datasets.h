#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include <string>
#include <vector>

#include "Tensor.h"

namespace SingleNet
{
    std::vector<unsigned char *> read_mnist_images(std::string full_path, int &number_of_images, int &image_size);
    std::vector<unsigned char> read_mnist_labels(std::string full_path, int &number_of_labels);

    template <size_t Batch, size_t ImageSize>
    std::vector<Tensor<float, Batch, ImageSize>> Image(std::string full_path)
    {
        int image_num, image_size;
        auto raw_images = read_mnist_images(full_path, image_num, image_size);

        std::vector<Tensor<float, Batch, ImageSize>> mnist_images;
        for (int i = 0; i < image_num / Batch; i++)
        {
            Tensor<float, Batch, ImageSize> temp;
            for (int j = 0; j < Batch; j++)
            {
                for (int k = 0; k < ImageSize; k++)
                    temp[j][k] = ((float)raw_images[i * Batch + j][k]) / 255.0f;
            }
            mnist_images.push_back(temp);
        }

        return mnist_images;
    }

    template <size_t Batch, size_t LabelSize>
    std::vector<Tensor<bool, Batch, LabelSize>> Label(std::string full_path)
    {
        int label_num;
        auto raw_labels = read_mnist_labels(full_path, label_num);

        std::vector<Tensor<bool, Batch, LabelSize>> mnist_labels;
        for (int i = 0; i < label_num / Batch; i++)
        {
            Tensor<bool, Batch, LabelSize> temp(false);
            for (int j = 0; j < Batch; j++)
                temp[j][raw_labels[i * Batch + j]] = true;
            mnist_labels.push_back(temp);
        }

        return mnist_labels;
    }
}

#endif