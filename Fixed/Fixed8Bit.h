#ifndef FIXED8BIT_H
#define FIXED8BIT_H

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

#endif