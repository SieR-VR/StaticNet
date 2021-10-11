#include "Datasets.h"

#include <fstream>
#include <algorithm>

std::vector<unsigned char *> SingleNet::Datasets::MNIST::Raw::read_mnist_images(std::string full_path, int &number_of_images, int &image_size)
{
    auto reverseInt = [](int i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open())
    {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051)
            throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        std::vector<unsigned char *> _dataset(number_of_images);
        for (int i = 0; i < number_of_images; i++)
        {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        file.close();
        return _dataset;
    }
    else
    {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

std::vector<unsigned char> SingleNet::Datasets::MNIST::Raw::read_mnist_labels(std::string full_path, int &number_of_labels)
{
    auto reverseInt = [](int i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open())
    {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049)
            throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        std::vector<unsigned char> _dataset(number_of_labels);
        for (int i = 0; i < number_of_labels; i++)
        {
            file.read((char *)&_dataset[i], 1);
        }
        file.close();
        return _dataset;
    }
    else
    {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

SingleNet::Vector<float, 2> SingleNet::Datasets::MNIST::Image(std::string full_path)
{
    int image_num, image_size;
    auto raw_images = Raw::read_mnist_images(full_path, image_num, image_size);

    SingleNet::Vector<float, 2> mnist_images;
    for (int i = 0; i < image_num; i++)
    {
        SingleNet::Vector<float, 1> temp;
        for (int j = 0; j < image_size; j++)
            temp.push_back(((int)raw_images[i][j]) / 255.0f);
        mnist_images.push_back(temp);
    }

    return mnist_images;
}

SingleNet::Vector<float, 2> SingleNet::Datasets::MNIST::Label(std::string full_path)
{
    int label_num;
    auto raw_labels = Raw::read_mnist_labels(full_path, label_num);

    SingleNet::Vector<float, 2> mnist_labels;
    for (int i = 0; i < label_num; i++)
    {
        SingleNet::Vector<float, 1> temp;
        for (int j = 0; j < 10; j++) // 10 classes
            temp.push_back(raw_labels[i] == j ? true : false);
        mnist_labels.push_back(temp);
    }

    return mnist_labels;
}

SingleNet::Vector<size_t, 1> SingleNet::Datasets::RandomIndexes(size_t size, size_t number_of_indexes)
{
    std::vector<size_t> indexes;
    for (size_t i = 0; i < size; i++)
        indexes.push_back(i);

    std::random_shuffle(indexes.begin(), indexes.end());

    SingleNet::Vector<size_t, 1> random_indexes;
    for (size_t i = 0; i < number_of_indexes; i++)
        random_indexes.push_back(indexes[i]);

    return random_indexes;
}