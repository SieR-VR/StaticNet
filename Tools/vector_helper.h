#ifndef VECTOR_HELPER_H
#define VECTOR_HELPER_H

#include <vector>
#include <memory>

template <typename T>
std::vector<T> vector_split(std::vector<T> input, int start, int end) {
    std::vector<T> res;
    for(int i = start; i < end; i++)
        res.push_back(input[i]);

    return res;
}

#endif