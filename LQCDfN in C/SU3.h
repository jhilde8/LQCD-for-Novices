#ifndef SU3_H
#define SU3_H

#include <complex>
#include <C:\Users\Hilde\LQCD\eigen-3.4.0\Eigen\Dense>
#include <iostream>
#include <random>
using namespace std;

namespace su3
{
    class SU3{
        public:
            typedef Eigen::Matrix3cf cfm3; 
            typedef Eigen::Matrix3cd cdm3;
            typedef complex<float> cf;
            typedef complex<double> cd;
            random_device rd;
            mt19937 mt;
            uniform_real_distribution<double> dist;

            cdm3 H; //fills H with random complex floats from -1 to 1. 
            cdm3 U; 
            cdm3 Udag;
            cdm3 Hdag;
            cdm3 A;
            int terms = 20;
            const float epsM = 0.24;
            cd c;
            cd D;
            cd eps = 0.0 + 0.24i;
            const int Nm = 100; 
            SU3();
            template<typename T> void MatPow(T &M,int p);
    };
};

#endif
