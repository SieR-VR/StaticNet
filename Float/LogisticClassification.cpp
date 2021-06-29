#include "LogisticClassification.h"
#include "../Tools/model_data_defines.h"

int LogisticClassification::logisticClassify(std::vector<float> input) {
    std::vector<float> logisticRegs(nodes.size());

    for(int i = 0; i < nodes.size(); i++) {
        logisticRegs[i] = nodes[i].logisticReg(input);
    }

    float res_float = -100; int index = 0; 
    for(int i = 0; i < nodes.size(); i++) {
        if(res_float < logisticRegs[i]) {
            res_float = logisticRegs[i];
            index = i;
        }
    }

    return index;
}

std::vector<float> LogisticClassification::softmax(std::vector<float> input) {
    float sumExp = 0;
    for(int i = 0; i < input.size(); i++)
        sumExp += exp(input[i]);

    std::vector<float> res(input.size());
    for(int i = 0; i < input.size(); i++)
        res[i] = exp(input[i]) / sumExp;

    return res;
}

float LogisticClassification::getCost(std::vector<std::vector<float>> inputs, std::vector<std::vector<bool>> results) {
    float res = 0;
    for(int i = 0; i < nodes.size(); i++)
        res += nodes[i].getCost(inputs, results[i]);

    return res;
}

float LogisticClassification::gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<std::vector<bool>> results) {
    float costSum = 0;

    for(int i = 0; i < nodes.size(); i++) {
        costSum += nodes[i].gradientDescent(alpha, inputs, results[i]);
    }

    return costSum;
}

std::vector<uint8_t> LogisticClassification::getModelData() {
    std::vector<uint8_t> modelData;

    modelData.push_back(static_cast<uint8_t>(modelNumberType::FLOAT));
    modelData.push_back(static_cast<uint8_t>(modelType::LOGISTIC_CLASSIFICATON));

    uint32_t modelSize = nodes.size();
    modelData.push_back(((uint8_t*)&modelSize)[0]);
    modelData.push_back(((uint8_t*)&modelSize)[1]);
    modelData.push_back(((uint8_t*)&modelSize)[2]);
    modelData.push_back(((uint8_t*)&modelSize)[3]);

    for(auto &node : nodes) {
        auto nodeData = node.getModelData();
        for(auto &k : nodeData) modelData.push_back(k);
    }

    return modelData;
}