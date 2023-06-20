#ifndef STAPLE_H
#define STAPLE_H

#include <iostream>
#include <stdio.h>
#include <complex>
#include <random>
#include <C:\Users\Hilde\LQCD\eigen-3.4.0\Eigen\Dense> 
#include "SLattice.h"
using Eigen::Matrix3cd;
using Eigen::Vector4i;

//class creates the staple object, whick is the sum of the product of link variables 
//that create the square or rectangle operators when combined with the link variable at x,y,z,t.
//this will be initialized with a specific location and direction, as we only want the staples connected
//to this specific link everytime we create a staple object.
namespace staple
{
    class Staple: private sL::Lattice
    {
        private:
            Vector4i mu_vec,nu_vec, loc, loc_mu,loc_nu,loc_mu_nu,loc_m_mu,loc_2mu,loc_2nu,loc_m_nu,loc_m_2nu,loc_m_mu_m_nu,loc_m_mu_nu,loc_mu_m_nu,loc_mu_m_2nu,loc_2mu_m_nu;
            Matrix3cd U1,U2,U3,U4;
        public:
            Matrix3cd G,O;
            Staple(int x,int y, int z, int t, int mu);
            void Gamma(int x,int y, int z, int t, int mu);
            void Omega(int x,int y, int z, int t, int mu);
            //~Staple();
    };
};

#endif