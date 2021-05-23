#include "Fixed8Bit.h"

fixed8bit fixed8bit::operator+(fixed8bit operand)
{
    if(((short) operand.value + value) > CHAR_MAX) return CHAR_MAX;
    else if(((short) operand.value + value) < CHAR_MIN) return CHAR_MIN;
    else return operand.value + value;
}

fixed8bit fixed8bit::operator+=(fixed8bit operand)
{
    if(((short) operand.value + value) > CHAR_MAX) value = CHAR_MAX;
    else if(((short) operand.value + value) < CHAR_MIN) value = CHAR_MIN;
    else value = operand.value + value;
}

fixed8bit fixed8bit::operator-(fixed8bit operand)
{
    if(((short) value - operand.value) > CHAR_MAX) return CHAR_MAX;
    else if(((short) value - operand.value) < CHAR_MIN) return CHAR_MIN;
    else return value - operand.value;
}

fixed8bit fixed8bit::operator-=(fixed8bit operand)
{
    if(((short) value - operand.value) > CHAR_MAX) value = CHAR_MAX;
    else if(((short) value - operand.value) < CHAR_MIN) value = CHAR_MIN;
    else value = value - operand.value;
}

fixed8bit fixed8bit::operator*(fixed8bit operand)
{
    short operResRaw = (short)value * operand.value;
    short sign = operResRaw & 0x8000;

    return (operResRaw >> 7) | (sign >> 8);
}

fixed8bit fixed8bit::operator/(char operand)
{
    return value / operand;
}

float fixed8bit::mean()
{
    return value * 0.0078125f;
}
