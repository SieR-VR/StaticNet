#ifndef MYNET_H
#define MYNET_H

#include <vector>
#include <limits.h>

struct fixed8bit {
    char value;

    fixed8bit() : value(0x00) {}
    fixed8bit(char val) : value(val) {}

    fixed8bit operator+(fixed8bit operand);
    fixed8bit operator+=(fixed8bit operand);
    fixed8bit operator-(fixed8bit operand);
    fixed8bit operator-=(fixed8bit operand);
    fixed8bit operator*(fixed8bit operand);
    fixed8bit operator/(char operand);

    float mean();
};

class MyFixNet {
public:
    std::vector<fixed8bit> W;
    fixed8bit b;

    MyFixNet(unsigned int argumentsNum) {
        W.assign(argumentsNum, fixed8bit(0x40));
        b = 0x00;
    }

    fixed8bit getCost(std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results);
    fixed8bit getCostDiff(std::vector<fixed8bit> inputs, fixed8bit result, unsigned char index);
    fixed8bit linearReg(std::vector<fixed8bit> input);
    fixed8bit gradientDescent(fixed8bit alpha, std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results);
};

#endif