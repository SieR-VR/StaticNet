#ifndef MYNET_H
#define MYNET_H

#include <vector>
#include <limits.h>

struct fixed8bit {
    char value;

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
    fixed8bit W, b;

    MyFixNet(fixed8bit mW = 0b01000000, fixed8bit mB = 0b00000000): W(mW), b(mB) {}

    fixed8bit getCost(std::vector<fixed8bit> inputs, std::vector<fixed8bit> results);
    fixed8bit getCostDiff(std::vector<fixed8bit> inputs, std::vector<fixed8bit> results);
    fixed8bit linearReg(fixed8bit input);
    fixed8bit gradientDescent(fixed8bit alpha, std::vector<fixed8bit> inputs, std::vector<fixed8bit> results);
};

#endif