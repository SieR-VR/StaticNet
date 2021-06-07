#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<unsigned char *> read_mnist_images(std::string full_path, int& number_of_images, int& image_size);
std::vector<unsigned char> read_mnist_labels(std::string full_path, int& number_of_labels);

#endif