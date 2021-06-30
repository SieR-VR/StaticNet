#ifndef SAVE_MODEL_DATA
#define SAVE_MODEL_DATA

#include <vector>
#include <stdint.h>
#include <fstream>
#include <stdexcept>
#include <string>

void saveModelData(std::vector<uint8_t> modelData, std::string filePath);
std::vector<uint8_t> loadModelDataFromFile(std::string filePath);

#endif