#ifndef MYNET_H
#define MYNET_H

#include <vector>

struct fixed8bit {
    uint8_t value;

    fixed8bit(uint8_t val) : value(val) {}

    fixed8bit operator+(fixed8bit operand) {
        bool firstOperandSign = value >> 7;
        uint8_t firstOperandAbs = (value & 0b01111111);

        bool secondOperandSign = operand.value >> 7;
        uint8_t secondOperandAbs = (operand.value & 0b01111111);

        if(!(firstOperandSign ^ secondOperandSign)) {
            if((uint16_t)firstOperandAbs + secondOperandAbs >= 128) return (firstOperandSign << 7 | 0b01111111);
            else return (firstOperandSign << 7 | (firstOperandAbs + secondOperandAbs));
        }
        else {
            if(firstOperandAbs > secondOperandAbs) return (firstOperandSign << 7 | (firstOperandAbs - secondOperandAbs));
            else return (secondOperandSign << 7 | (secondOperandAbs - firstOperandAbs));
        }
    }

    fixed8bit operator+=(fixed8bit operand) {
        bool firstOperandSign = value >> 7;
        uint8_t firstOperandAbs = (value & 0b01111111);

        bool secondOperandSign = operand.value >> 7;
        uint8_t secondOperandAbs = (operand.value & 0b01111111);

        if(!(firstOperandSign ^ secondOperandSign)) {
            if((uint16_t)firstOperandAbs + secondOperandAbs >= 128) value = (firstOperandSign << 7 | 0b01111111);
            else value = (firstOperandSign << 7 | (firstOperandAbs + secondOperandAbs));
        }
        else {
            if(firstOperandAbs > secondOperandAbs) value = (firstOperandSign << 7 | (firstOperandAbs - secondOperandAbs));
            else value = (secondOperandSign << 7 | (secondOperandAbs - firstOperandAbs));
        }
    }

    fixed8bit operator-(fixed8bit operand) {
        bool secondOperandSign = operand.value >> 7;
        uint8_t secondOperandAbs = (operand.value & 0b01111111);
        return this->operator+((!secondOperandSign << 7) | secondOperandAbs);
    }

    fixed8bit operator-=(fixed8bit operand) {
        bool secondOperandSign = operand.value >> 7;
        uint8_t secondOperandAbs = (operand.value & 0b01111111);
        this->operator+=((!secondOperandSign << 7) | secondOperandAbs);
    }

    fixed8bit operator*(fixed8bit operand) {
        bool firstOperandSign = value >> 7;
        uint8_t firstOperandAbs = (value & 0b01111111);

        bool secondOperandSign = operand.value >> 7;
        uint8_t secondOperandAbs = (operand.value & 0b01111111);

        uint8_t operResult = ((uint16_t)firstOperandAbs * secondOperandAbs) >> 7;
        if(firstOperandSign ^ secondOperandSign) return (1 << 7 | operResult);
        else return operResult;
    }

    fixed8bit operator/(uint8_t operand) {
        bool sign = value >> 7;
        uint8_t operandAbs = (value & 0b01111111); 

        uint8_t operResult = operandAbs / operand;
        if(sign) return (1 << 7 | operResult);
        else return operResult;
    }

    float mean() {
        bool sign = value >> 7;
        float result = (value & 0b01111111) * 0.0078125f;

        if(sign) return -1 * result;
        else return result;
    }
};

class MyFixNet {
public:
    fixed8bit W = 0b01000000, b = 0b00000000;

    fixed8bit getCost(std::vector<fixed8bit> inputs, std::vector<fixed8bit> results) {
        if(inputs.size() != results.size()) return 0;
        fixed8bit resultCost = 0;

        for(int i = 0; i < inputs.size(); i++) {
            fixed8bit cost = linearReg(inputs[i]) - results[i];
            resultCost += cost * cost;
        }

        return resultCost / inputs.size();
    }

    fixed8bit getCostDiff(std::vector<fixed8bit> inputs, std::vector<fixed8bit> results) {
        if(inputs.size() != results.size()) return 0;
        fixed8bit resultCostDiff = 0;

        for(int i = 0; i < inputs.size(); i++) {
            fixed8bit costDiff = (linearReg(inputs[i]) - results[i]) * inputs[i];
            resultCostDiff += costDiff;
        }

        return resultCostDiff / inputs.size();
    }

    fixed8bit linearReg(fixed8bit input) {
        return W * input + b;
    }

    fixed8bit gradientDescent(fixed8bit alpha, std::vector<fixed8bit> inputs, std::vector<fixed8bit> results) {
        W -= getCostDiff(inputs, results) * alpha;
    }
};

#endif