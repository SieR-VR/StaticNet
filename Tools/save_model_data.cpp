#include "save_model_data.h"

void saveModelData(Vector1D<uint8_t> modelData, std::string filePath) {
    std::ofstream file(filePath, std::ios::binary);
    file.write((char *)modelData.value.data(), modelData.shape().x);
    file.close();
}

Vector1D<uint8_t> loadModelDataFromFile(std::string filePath) {
    std::ifstream file(filePath, std::ios::binary);

    if(!file.is_open())
        throw std::runtime_error("Invalid file name!");

    file.seekg(0, std::ios::end);
    int length = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> modelData(length);
    file.read((char *)modelData.data(), length);

    file.close();

    return modelData;
}