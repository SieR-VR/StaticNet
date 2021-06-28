#ifndef MODEL_DATA_DEFINES
#define MODEL_DATA_DEFINES

#include <stdint.h>

enum class modelNumberType: uint8_t
{
    FIXED = 0,
    FLOAT = 1
};

enum class modelType: uint8_t
{
    LINEAR_REGRESSION = 0,
    LOGISTIC_REGRESSION = 1,
    LOGISTIC_CLASSIFICATON = 2
};

#endif