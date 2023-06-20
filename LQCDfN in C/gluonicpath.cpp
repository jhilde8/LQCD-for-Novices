#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <fstream>
#include "SU3.h"
#include "SLattice.h"
using namespace std;
using namespace sL; 

int main(){
    ofstream myfile;
    myfile.open("output.txt");

    Lattice Lat; 
    Eigen::Matrix3cd N;
    Lat.Update();
    N = Lat.L[0][0][0][0][0];

    float d = real(N.trace());
    //cout << d << endl;
    myfile << d << flush;
    myfile.close();

    return 0;
}
