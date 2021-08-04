#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include <fstream>
#include <string>
#include "../Structure/Vector.h"

namespace SingleNet
{
    namespace Datasets
    {
        namespace MNIST
        {
            namespace Raw
            {
                std::vector<unsigned char *> read_mnist_images(std::string full_path, int &number_of_images, int &image_size);
                std::vector<unsigned char> read_mnist_labels(std::string full_path, int &number_of_labels);
            }

            Vector<float, 2> Image(std::string full_path);
            Vector<bool, 2> Label(std::string full_path);
        }
    }
}

#endif