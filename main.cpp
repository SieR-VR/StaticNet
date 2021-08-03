#include <iostream>

#include "Structure/Vector.h"

using namespace std;

int main(int argc, char *argv[])
{
    try
    {
        Vector<float, 1> v1 = Vector<float, 1>();
        Vector<float, 1> v2 = Vector<float, 1>();

        v1.push_back(1.0f);
        v1.push_back(2.0f);
        v1.push_back(3.0f);

        v2.push_back(4.0f);
        v2.push_back(5.0f);
        v2.push_back(6.0f);

        Vector<float, 2> v3 = Vector<float, 2>();
        Vector<float, 2> v4 = Vector<float, 2>();

        v3.push_back(v1);
        v3.push_back(v2);
        v3.push_back(v1);

        v4.push_back(v2);
        v4.push_back(v1);
        v4.push_back(v2);

        cout << "v1 = " << v1 << endl;
        cout << "v2 = " << v2 << endl;

        cout << "v3 = " << v3 << endl;
        cout << "v4 = " << v4 << endl;

        cout << "v3 + v4 = " << v3 + v4 << endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}