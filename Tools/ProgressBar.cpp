#include "ProgressBar.h"

ProgressBar::ProgressBar(int maxValue)
{
    m_maxValue = maxValue;
    m_value = 0;
    m_visibleFlag = false;
}

ProgressBar::~ProgressBar()
{
}

void ProgressBar::update(int value, const std::string &text)
{
    if(value > m_maxValue)
        value = m_maxValue;
    
    if(value < 0)
        value = 0;

    std::string msg = "";
    msg += "[";

    float percentage = (float)value / (float)m_maxValue * 100.0f;
    int progress = (int)(percentage / 2.0f);
    for (int i = 0; i < 50; i++)
    {
        if (i < progress) msg += "#";
        else msg += " ";
    }

    msg += "] ";
    msg += text;
    std::cout << msg << "\r";
}
            
            

    