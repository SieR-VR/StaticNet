#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Fixed/Fixed8Bit.h"

std::vector<unsigned char *> read_mnist_images(std::string full_path, int& number_of_images, int& image_size);
std::vector<unsigned char> read_mnist_labels(std::string full_path, int& number_of_labels);

std::vector<std::vector<float>> get_mnist_image_float(std::string full_path);
std::vector<std::vector<fixed8bit>> get_mnist_image_fixed(std::string full_path);
std::vector<std::vector<bool>> get_mnist_label(std::string full_path);

#endif