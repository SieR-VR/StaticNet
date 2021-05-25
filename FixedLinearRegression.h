#ifndef FIXEDLINEARREGRESSION_H
#define FIXEDLINEARREGRESSION_H

#include <vector>
#include "Fixed8Bit.h"

class FixedLinearRegression {
public:
    std::vector<fixed8bit> W;
    fixed8bit b;

    FixedLinearRegression(unsigned int argumentsNum) {
        W.assign(argumentsNum, fixed8bit(0x00));
        b = 0x00;
    }

    fixed8bit getCost(std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results);
    fixed8bit getCostDiff(std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results, unsigned char index);
    fixed8bit getBiasDiff(std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results);
    fixed8bit linearReg(std::vector<fixed8bit> input);
    fixed8bit sigmoid(std::vector<fixed8bit> input);
    fixed8bit gradientDescent(fixed8bit alpha, std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results);
};

#endif