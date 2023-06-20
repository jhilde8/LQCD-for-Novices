//To avoid having to compile 3000 files from the terminal, I think I can write a series of header files
//then only have one cpp file with all functionality from all of those headers included.
//that way I only have to compile 2 files (or 3 maybe idk) at compile time, the cpp
//with all of the functions at whatnot from the headers defined, then the main cpp file. 

#include <iostream>
#include <complex>
#include <random>
#include <ctime>
#include <C:\Users\Hilde\LQCD\eigen-3.4.0\Eigen\Dense>
#include "SU3.h"
#include "SLattice.h"
#include "staple.h"
using namespace std;
using namespace su3; 
using namespace sL;
using namespace staple;
using Eigen::Matrix3cd;

SU3::SU3(){
    mt19937 mt(rd());
    uniform_real_distribution<float> dist(-1,1);

    for (int j = 0; j<3;j++){
        for (int k = 0; k < 3; k++){
            double r = dist(mt);
            double c = dist(mt);

            cd m(r,c);
            H(j,k) = m; //This is a single 3x3 matrix filled with random complex numbmers.

        }

    }
    //H is fine at this point
    Hdag = H;
    Hdag.adjointInPlace();

    H = (H + Hdag)/2; //H is now Hermitian. //this seems to work 

    for (int p = 0; p < terms; p++){
        A = H;
        c = (pow(eps,p))/(tgamma(p+1));
        MatPow(A,p);
        A = c*A;

        U += A;
    }

    D = U.determinant();
    U = U/(pow(D,1/3));
    Udag = U;
    Udag.adjointInPlace();

}

template<typename T> void SU3::MatPow(T &M,int p){
    T I = T::Identity();
    T Mnew = M;

    if(p == 0){
        M = T::Identity();
    }else if(p == 1){
        M = T::Identity()*M;
    }else{
        MatPow(Mnew,p-1);
        M = M*Mnew;
    }

}

Lattice::Lattice(){
    
    for (int x = 0; x<N;x++){
        for(int y = 0; y<N;y++){
            for(int z = 0; z<N;z++){
                for(int t = 0; t<N;t++){
                    for(int mu = 0; mu<4;mu++){
                        L[x][y][z][t][mu] = Matrix3cd::Identity();
                    }
                }
            }
        }
    }

    GetSU3();

}

//constructs the array of random SU3 matricies and their adjoints
void Lattice::GetSU3(){
    //Matrix3cd M[200];
    for(int p = 0; p<Nm; p++){
        SU3 m;
        M[p] = m.U;
        M[p+Nm] = m.Udag;
    }

}

void Lattice::Update(){
    mt19937 up(rd());
    uniform_int_distribution<int> m_index(0,2*Nm);
    uniform_real_distribution<float> u_roll(0,1);
    float roll = u_roll(up);

    for (int x = 0; x<N;x++){
        for(int y = 0; y<N;y++){
            for(int z = 0; z<N;z++){
                for(int t = 0; t<N;t++){
                    for(int mu = 0; mu<4;mu++){
                        Staple S(x,y,z,t,mu); 
                        //Gamma(x,y,z,t,mu);
                        //Lambda(x,y,z,t,mu);
                        //Matrix3cd G = gamma[x][y][z][t][mu];
                        //Matrix3cd R = omega[x][y][z][t][mu];

                        for (int h = 0; h<hits;h++){
                            int ind = m_index(up);
                            Lnew = M[ind]*L[x][y][z][t][mu];
                            B = (Lnew - L[x][y][z][t][mu])*S.G;
                            C = (Lnew - L[x][y][z][t][mu])*S.O;

                            if (improved == true){
                                
                                dS = (-beta_imp/3)*((5/(3*u0*u0*u0*u0))*real(B.trace()) - (1/(12*u0*u0*u0*u0*u0*u0))*real(C.trace()));

                            } else {
                         
                                dS = (-beta/3)*real(B.trace());

                            }

                            if (dS < 0 || roll < exp(-dS)){
                                L[x][y][z][t][mu] = Lnew;
                            } else {
                                L[x][y][z][t][mu] = L[x][y][z][t][mu];
                            }

                            //we calculate the action with the new U, improved and unimproved

                            //if this change in action is less that 0 or if a random number we have is less than e^(-dS), we keep the change.
                            //if not, we keep the old U.  
                        }
                    }
                }
            }
        }
    }
}

Staple::Staple(int x,int y,int z, int t, int mu){
    mu_vec,nu_vec = {0,0,0,0};
    loc = {x,y,z,t};
    loc_mu,loc_nu,loc_mu_nu,loc_m_mu,loc_2mu,loc_2nu,loc_m_nu,loc_m_2nu,loc_m_mu_m_nu,loc_m_mu_nu,loc_mu_m_nu,loc_mu_m_2nu,loc_2mu_m_nu = loc;
    mu_vec(mu) = 1;
    loc_mu(mu) = (loc(mu) + mu_vec(mu))%N;
    loc_2mu(mu) = (loc_mu(mu) + mu_vec(mu))%N;
    loc_m_mu(mu) = (loc(mu) - mu_vec(mu))%N;

    U1,U2,U3,U4 = Matrix3cd::Zero();

    Gamma(x,y,z,t,mu);
    Omega(x,y,z,t,mu);

}

//Function to calculate the staple of a given link variable. This is called in update,
//as in the action we have the square operator, requiring each time we calculate S, we calculate a staple.
//we want to pass in a location and Lorentz index, then edit the matrix representing the staple to be the product of link variables that make it up. 
void Staple::Gamma(int x,int y, int z, int t, int ind){

    G = Matrix3cd::Zero();
    
    for(int nu = 0; nu<4; nu++){
        if (nu != ind){
            loc_nu,loc_mu_nu,loc_mu_m_nu,loc_m_nu = loc;
            nu_vec(nu) = 1;
            loc_nu(nu) = (loc(nu) + nu_vec(nu))%N;
            loc_mu_nu(nu) = (loc(nu) + nu_vec(nu)) % N;
            loc_mu_nu(ind) = (loc(ind) + mu_vec(ind)) %N;
            loc_m_nu(nu) = (loc(nu) - nu_vec(nu))%N;
            loc_mu_m_nu(nu) = (loc(nu) - nu_vec(nu)) % N;
            loc_mu_m_nu(ind) = (loc(ind) + mu_vec(ind)) %N; 

            U2 = L[loc_nu(0)][loc_nu(1)][loc_nu(2)][loc_nu(3)][ind];
            U2.adjointInPlace();
            U1 = L[loc(0)][loc(1)][loc(2)][loc(3)][nu];
            U1.adjointInPlace();
            U3 = L[loc_mu_m_nu(0)][loc_mu_m_nu(1)][loc_mu_m_nu(2)][loc_mu_m_nu(3)][nu];
            U3.adjointInPlace();
            U4 = L[loc_m_nu(0)][loc_m_nu(1)][loc_m_nu(2)][loc_m_nu(3)][ind];
            U4.adjointInPlace();

            G += L[loc_mu(0)][loc_mu(1)][loc_mu(2)][loc_mu(3)][nu]*U2*U1;
            G += U3*U4*L[loc_m_nu(0)][loc_m_nu(1)][loc_m_nu(2)][loc_m_nu(3)][nu];
            // gamma +=  self.U[loc_mu[0],loc_mu[1],loc_mu[2],loc_mu[3],nu]@self.dagger(self.U[loc_nu[0],loc_nu[1],loc_nu[2],loc_nu[3],ind])@self.dagger(self.U[loc[0],loc[1],loc[2],loc[3],nu]) #positive nu product
            //  gamma +=  self.dagger(self.U[loc_mu_n_nu[0],loc_mu_n_nu[1],loc_mu_n_nu[2],loc_mu_n_nu[3],nu])@self.dagger(self.U[loc_n_nu[0],loc_n_nu[1],loc_n_nu[2],loc_n_nu[3],ind])@self.U[loc_n_nu[0],loc_n_nu[1],loc_n_nu[2],loc_n_nu[3],nu]
            nu_vec(nu) = 0;
        }

    }
    



}


void Staple::Omega(int x,int y, int z, int t, int mu){
   
    O = Matrix3cd::Zero(); 

    for(int nu = 0; nu < 4; nu++){
        if (nu != mu){
            loc_nu,loc_mu_nu,loc_2nu,loc_m_nu,loc_m_2nu,loc_m_mu_m_nu,loc_m_mu_nu,loc_mu_m_nu,loc_mu_m_2nu,loc_2mu_m_nu = loc;
            nu_vec(nu) = 1;

            loc_nu(nu) = (loc_nu(nu) + nu_vec(nu))%N;
            loc_m_nu(nu) = (loc_nu(nu) - nu_vec(nu))%N;
            loc_m_2nu(nu) = (loc_m_nu(nu) - nu_vec(nu))%N;
            loc_2nu(nu) = (loc_nu(nu) + nu_vec(nu))%N;

            loc_mu_nu(mu) = loc_mu(mu);
            loc_mu_nu(nu) = loc_nu(nu);
            loc_m_mu_m_nu(mu) = loc_m_mu(mu);
            loc_m_mu_m_nu(nu) = loc_m_nu(nu);
            loc_m_mu_nu(mu) = loc_m_mu(mu);
            loc_m_mu_nu(nu) = loc_nu(nu);
            loc_mu_m_nu(mu) = loc_mu(mu);
            loc_mu_m_nu(nu) = loc_m_nu(nu);
            loc_mu_m_2nu(mu) = loc_mu(mu);
            loc_mu_m_2nu(nu) = loc_m_2nu(nu);
            loc_2mu_m_nu(mu) = loc_2mu(mu);
            loc_2mu_m_nu(nu) = loc_m_nu(nu);
            
            U1 = L[loc_2nu(0)][loc_2nu(1)][loc_2nu(2)][loc_2nu(3)][mu]; //loc_2nu in mu
            U2 = L[loc_nu(0)][loc_nu(1)][loc_nu(2)][loc_nu(3)][nu]; //loc_nu in nu
            U3 = L[x][y][z][t][nu]; //loc in nu
            U1.adjointInPlace(); 
            U2.adjointInPlace(); 
            U3.adjointInPlace(); 
            O += L[loc_mu(0)][loc_mu(1)][loc_mu(2)][loc_mu(3)][nu]*L[loc_mu_nu(0)][loc_mu_nu(1)][loc_mu_nu(2)][loc_mu_nu(3)][nu]*U1*U2*U3;

            U1 = L[loc_nu(0)][loc_nu(1)][loc_nu(2)][loc_nu(3)][mu]; //loc_2nu in mu
            U2 = L[loc_m_mu_nu(0)][loc_m_mu_nu(1)][loc_m_mu_nu(2)][loc_m_mu_nu(3)][mu]; //loc_nu in nu
            U3 = L[loc_m_mu(0)][loc_m_mu(1)][loc_m_mu(2)][loc_m_mu(3)][nu]; //loc in nu
            U1.adjointInPlace(); 
            U2.adjointInPlace(); 
            U3.adjointInPlace(); 
            O+= L[loc_mu(0)][loc_mu(1)][loc_mu(2)][loc_mu(3)][nu]*U1*U2*U3*L[loc_m_mu(0)][loc_m_mu(1)][loc_m_mu(2)][loc_m_mu(3)][mu];

            U1 = L[loc_mu_nu(0)][loc_mu_nu(1)][loc_mu_nu(2)][loc_mu_nu(3)][mu]; //loc_2nu in mu
            U2 = L[loc_nu(0)][loc_nu(1)][loc_nu(2)][loc_nu(3)][mu]; //loc_nu in nu
            U3 = L[x][y][z][t][nu]; //loc in nu
            U1.adjointInPlace(); 
            U2.adjointInPlace(); 
            U3.adjointInPlace();
            O += L[loc_mu(0)][loc_mu(1)][loc_mu(2)][loc_mu(3)][mu]*L[loc_2mu(0)][loc_2mu(0)][loc_2mu(0)][loc_2mu(0)][nu]*U1*U2*U3;

            U1 = L[loc_2mu_m_nu(0)][loc_2mu_m_nu(1)][loc_2mu_m_nu(2)][loc_2mu_m_nu(3)][nu]; //loc_2nu in mu
            U2 = L[loc_mu_m_nu(0)][loc_mu_m_nu(1)][loc_mu_m_nu(2)][loc_mu_m_nu(3)][mu]; //loc_nu in nu
            U3 = L[loc_m_nu(0)][loc_m_nu(1)][loc_m_nu(2)][loc_m_nu(3)][mu]; //loc in nu
            U1.adjointInPlace(); 
            U2.adjointInPlace(); 
            U3.adjointInPlace();
            O += L[loc_mu(0)][loc_mu(1)][loc_mu(2)][loc_mu(3)][mu]*U1*U2*U3*L[loc_m_nu(0)][loc_m_nu(1)][loc_m_nu(2)][loc_m_nu(3)][nu];

            U1 = L[loc_mu_m_nu(0)][loc_mu_m_nu(1)][loc_mu_m_nu(2)][loc_mu_m_nu(3)][nu]; //loc_nu in nu
            U2 = L[loc_mu_m_2nu(0)][loc_mu_m_2nu(1)][loc_mu_m_2nu(2)][loc_mu_m_2nu(3)][nu]; //loc_2nu in mu
            U3 = L[loc_m_2nu(0)][loc_m_2nu(1)][loc_m_2nu(2)][loc_m_2nu(3)][mu]; //loc in nu
            U1.adjointInPlace(); 
            U2.adjointInPlace(); 
            U3.adjointInPlace();
            O += U1*U2*U3*L[loc_m_2nu(0)][loc_m_2nu(1)][loc_m_2nu(2)][loc_m_2nu(3)][nu]*L[loc_m_nu(0)][loc_m_nu(1)][loc_m_nu(2)][loc_m_nu(3)][nu];

            U1 = L[loc_mu_m_nu(0)][loc_mu_m_nu(1)][loc_mu_m_nu(2)][loc_mu_m_nu(3)][nu]; //loc_nu in nu
            U2 = L[loc_m_nu(0)][loc_m_nu(1)][loc_m_nu(2)][loc_m_nu(3)][mu];
            U3 = L[loc_m_mu_m_nu(0)][loc_m_mu_m_nu(1)][loc_m_mu_m_nu(2)][loc_m_mu_m_nu(3)][mu];
            U1.adjointInPlace(); 
            U2.adjointInPlace(); 
            U3.adjointInPlace();
            O += U1*U2*U3*L[loc_m_mu_m_nu(0)][loc_m_mu_m_nu(1)][loc_m_mu_m_nu(2)][loc_m_mu_m_nu(3)][nu]*L[loc_m_mu(0)][loc_m_mu(1)][loc_m_mu(2)][loc_m_mu(3)][mu];
        
            nu_vec(nu) = 0;
        }
    }

}

void Lattice::Thermalize(){
    for (int t = 0; t < 2*Ncor; t++){
        Update();
    }
}

void Lattice::Corr(){
    for (int t = 0; t < Ncor; t++){
        Update();
    }
}

//Function that evaluates all axa wilson loops on the lattice at a certain location. 
float Lattice::Wilsonaxa(int x,int y, int z, int t){
    float WL = 0;
    complex<float> temp;
    Eigen::Vector4i mu_vec,nu_vec,loc,loc_mu,loc_nu = Eigen::Vector4i::Zero();
    Matrix3cd U1,U2,W = Matrix3cd::Zero();

    loc = {x,y,z,t};

    for(int mu = 0; mu<4; mu++){
        loc_mu = loc;
        mu_vec(mu) = 1;
        loc_mu(mu) = (loc(mu) + mu_vec(mu))%N;
        for(int nu = 0; nu < mu; nu++){
            loc_nu = loc;
            nu_vec(nu) = 1;
            loc_nu(nu) = (loc(nu) + nu_vec(nu))%N;

            U1 = L[loc_nu(0)][loc_nu(1)][loc_nu(2)][loc_nu(3)][mu];
            U1.adjointInPlace();
            U2 = L[loc(0)][loc(1)][loc(2)][loc(3)][nu];
            U2.adjointInPlace();

            W = L[loc(0)][loc(1)][loc(2)][loc(3)][mu]*L[loc_mu(0)][loc_mu(1)][loc_mu(2)][loc_mu(3)][nu]*U1*U2;
            temp+= W.trace();
            nu_vec(nu) = 0;
        }
        mu_vec(mu) = 0;
    }

    WL = (1/(3*6))*real(temp);

    return WL;
}

float Lattice::Wilsonax2a(int x,int y, int z, int t){
    float WL = 0;
    complex<float> temp = 0;
    Eigen::Vector4i mu_vec,nu_vec,loc,loc_mu,loc_nu,loc_mu_nu,loc_2nu = Eigen::Vector4i::Zero();
    Matrix3cd U1,U2,U3,W = Matrix3cd::Zero();

    loc = {x,y,z,t}; 


    for(int mu = 0; mu<4; mu++){
        loc_mu = loc;
        mu_vec(mu) = 1;
        loc_mu(mu) = (loc(mu) + mu_vec(mu))%N;
        
        for(int nu = 0; nu < mu; nu++){
            loc_nu,loc_mu_nu,loc_2nu = loc;
            nu_vec(nu) = 1;

            loc_nu(nu) = (loc(nu) + nu_vec(nu))%N;
            loc_2nu(nu) = (loc_nu(nu) + nu_vec(nu))%N;
            loc_mu_nu(mu) = loc_mu(mu); 
            loc_mu_nu(nu) = loc_nu(nu); 

            U1 = L[loc_2nu(0)][loc_2nu(1)][loc_2nu(2)][loc_2nu(3)][mu];
            U1.adjointInPlace();
            U2 = L[loc_nu(0)][loc_nu(1)][loc_nu(2)][loc_nu(3)][nu];
            U2.adjointInPlace();
            U3 = L[loc(0)][loc(1)][loc(2)][loc(3)][nu];
            U3.adjointInPlace();

            W = L[loc(0)][loc(1)][loc(2)][loc(3)][mu]*L[loc_mu(0)][loc_mu(1)][loc_mu(2)][loc_mu(3)][nu]*L[loc_mu_nu(0)][loc_mu_nu(1)][loc_mu_nu(2)][loc_mu_nu(3)][nu]*U1*U2*U3;
            temp+= W.trace();
            nu_vec(nu) = 0; 
            loc_nu(nu),loc_mu_nu(nu),loc_2nu(nu) = loc(nu);
        }
        mu_vec(mu) = 0;
        loc_mu(mu) = loc(mu);
    }

    WL = (1/(3*6))*real(temp);

    return WL;
}



