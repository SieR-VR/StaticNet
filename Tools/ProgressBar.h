#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <iostream>
#include <string>
#include <cmath>

class ProgressBar
{
public:
    ProgressBar(int maxValue);
    ~ProgressBar();

    void update(int value, const std::string &text);

private:
    int m_maxValue;
    int m_value;
    std::string m_text;

    bool m_visibleFlag;
};

#endif