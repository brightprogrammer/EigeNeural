
// main.cpp 
  
// don't forget to include out neural network 
#include "NeuralNetwork.hpp" 
  
//... data generator code here 
void genData(std::string filename) 
{ 
    std::ofstream file1(filename + "-in"); 
    std::ofstream file2(filename + "-out"); 
    for (uint r = 0; r < 1000; r++) { 
        Scalar x = rand() / Scalar(RAND_MAX); 
        Scalar y = rand() / Scalar(RAND_MAX); 
        file1 << x << ", " << y << std::endl; 
        file2 << 2 * x + 10 + y << std::endl; 
    } 
    file1.close(); 
    file2.close(); 
} 

  
typedef std::vector<RowVector*> data; 
int main() 
{ 
    NeuralNetwork n({ 2, 3, 1 }); 
    data in_dat, out_dat; 
    genData("test"); 
    ReadCSV("test-in", in_dat); 
    ReadCSV("test-out", out_dat); 
    n.train(in_dat, out_dat); 
    return 0; 
} 