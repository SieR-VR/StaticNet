#ifndef STACK_TRACE_H
#define STACK_TRACE_H

#include <vector>
#include <string>
#include <iostream>

namespace SingleNet
{
    namespace Tools
    {
        class StackTrace
        {
        public:
            StackTrace(const std::string &msg);
            void print() const;
            std::string msg;
            std::vector<void*> m_stackTrace;
        };
    }
}

#endif