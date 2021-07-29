#ifndef SAVE_MODEL_DATA
#define SAVE_MODEL_DATA

#include <vector>
#include <stdint.h>
#include <fstream>
#include <stdexcept>
#include <string>

#include "../Structure/Vector1D.h"

void saveModelData(Vector1D<uint8_t> modelData, std::string filePath);
Vector1D<uint8_t> loadModelDataFromFile(std::string filePath);

#endif