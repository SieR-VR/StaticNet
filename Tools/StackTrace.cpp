#include "StackTrace.h"
#include <execinfo.h>

#ifdef __linux__
#include <linux/kernel.h>
#endif

SingleNet::Tools::StackTrace::StackTrace(const std::string &msg)
{
    void *array[100];
    char **strings;

    int size = backtrace(array, 100);

    for(int i = 1; i < size; i++)
        m_stackTrace.push_back(array[i]);

    this->msg = msg;
}

void SingleNet::Tools::StackTrace::print() const
{
#ifdef __linux__
    for (int i = 0; i < m_stackTrace.size(); i++)
        printk("[%d] %pF : %p\n", i, m_stackTrace[i], m_stackTrace[i]);
    printk("%s\n", msg.c_str());
#else
    for (int i = 0; i < m_stackTrace.size(); i++)
        printf("[%d] %p\n", i, m_stackTrace[i]);
    printf("%s\n", msg.c_str());
#endif
}