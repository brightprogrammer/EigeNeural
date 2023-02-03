
// main.cpp 
  
// don't forget to include out neural network 
#include "NeuralNetwork.hpp"

#include <fstream>
  
//... data generator code here 
void genData(std::string filename) 
{ 
    std::ofstream file1(filename + "-in"); 
    std::ofstream file2(filename + "-out"); 
    for (uint r = 0; r < 1000; r++) { 
        Scalar x = rand() / Scalar(RAND_MAX); 
        Scalar y = rand() / Scalar(RAND_MAX); 
        file1 << x << "," << y << std::endl;
        file2 << 2 * x + 10 + y << std::endl; 
    } 
    file1.close(); 
    file2.close(); 
} 

int main() {
    // create NN
    NeuralNetwork n({ 2, 3, 1 });
    n.learningRate = 0.005;

    // create sample data and read
    Data in_dat, out_dat;
    genData("test"); 
    ReadCSV("test-in", in_dat); 
    ReadCSV("test-out", out_dat);

    // train
    n.train(in_dat, out_dat); 

    return 0;
}
