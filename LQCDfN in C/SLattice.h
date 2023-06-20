#ifndef SLATTICE_H
#define SLATTICE_H

#include <iostream>
using namespace std;
#include <stdio.h>
#include <complex>
#include <random>
#include "SU3.h"
using namespace su3;
#include <C:\Users\Hilde\LQCD\eigen-3.4.0\Eigen\Dense> 
using Eigen::Matrix3cd;

namespace sL
{
    class Lattice{ 
        private:
            const float a = 0.25;
            const float beta = 5.5;
            const float beta_imp = 1.719;
            const float u0 = 797;
            const int hits = 10;
            const int Nm = 100;
            const int Nmfull = 200;
            //int Ncor = 50;

        public:
            const int Ncor = 50;
            const int Ncf = 10;
            const int N=4;
            bool improved;
            double dS;
            random_device rd;
            mt19937 mt;
            uniform_int_distribution<int> m_index;
            uniform_real_distribution<float> u_roll;
            Matrix3cd L[4][4][4][4][4]; 
            Matrix3cd B; //temporary link variable array that we can play around with when doing products of link variables
            Matrix3cd C;
            Matrix3cd Lnew;
            //Matrix3cd G; //matrix to hold the product of link variables for a specific staple
            //Matrix3cd O; //same as above but for the improved action
            Matrix3cd M[200];
            //we need to define the array of link variables here, 
            // and empty gamma and lambda matricies
            Lattice(); 
            void GetSU3();
            void Update();
            //void Gamma(int x,int y, int z, int t, int ind); //staple function for unimproved action
            //void Lambda(int x,int y, int z, int t, int ind); //staple function for improved action
            void Thermalize(); //function that thermalizes the lattice
            void Corr(); //function that updates the lattice Ncor times in between measurements
            void GaugeCov(); //function to calculate gauge covariant derivative. it might be more useful to put this in smear
            void Smear(); //function to smear the spatial variables on the lattice. this is used in the static quark exercise
            float Wilsonaxa(int x,int y, int z, int t);
            float Wilsonax2a(int x,int y, int z, int t);
    };


};

#endif 

//The idea here is that this class is the lattice. 
//We will have a contstructor that defines the lattice as the identity for each position and Lorentz index
//I think this will just use a Matrix from Eigen

//We need a link variable (3x3 matrix of complex doubles) for each position and Lorentz index. 
//not sure how to do this at the moment 
//a multidimensional array of such objects seems to be the least tedious way to do this














