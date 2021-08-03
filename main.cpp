#include <iostream>

#include "Structure/Vector.h"

using namespace std;

int main(int argc, char *argv[])
{
    Vector<float, 2> v1 = Vector<float, 2>();
    v1.resize(Vector<size_t, 1>({3, 4}), 1.0);

    Vector<float, 2> v2 = Vector<float, 2>();
    v2.resize(Vector<size_t, 1>({4, 2}), 2.0);

    cout << "v1: " << v1 << endl;
    cout << "v2: " << v2 << endl;
}