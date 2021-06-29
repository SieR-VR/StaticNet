#include "save_model_data.h"

void saveModelData(std::vector<uint8_t> modelData, std::string filePath) {
    std::ofstream file(filePath, std::ios::binary);
    file.write((char *)modelData.data(), modelData.size());
    file.close();
}

std::vector<uint8_t> loadModelData(std::string filePath) {
    std::ifstream file(filePath, std::ios::binary);

    file.seekg(0, std::ios::end);
    int length = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> modelData(length);
    file.read((char *)modelData.data(), length);

    file.close();

    return modelData;
}