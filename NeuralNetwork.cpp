#include "NeuralNetwork.hpp"
#include <fstream>

Scalar ReLU(Scalar x) {
    return x >= 0 ? x : 0;
}

Scalar ReLU_derivative(Scalar x) {
    return x >= 0 ? 1 : 0;
}

Scalar activationFunction(Scalar x) {
    return ReLU(x);
}

Scalar activationFunctionDerivative(Scalar x) {
    return ReLU_derivative(x);
}

// constructor of neural network class 
NeuralNetwork::NeuralNetwork(const Topology& topology, Scalar learningRate) {
    this->topology = topology;
    this->learningRate = learningRate; 
    for (uint i = 0; i < topology.size(); i++) {
        // initialze neuron layers 
        if (i == topology.size() - 1) {
            // output layer has no bias neurons
            neuronLayers.push_back(new RowVector(topology[i])); 
        } else {
            // all hidden layers and input layer have bias neuron
            neuronLayers.push_back(new RowVector(topology[i] + 1)); 
        }

        // initialize cache and delta vectors 
        cacheLayers.push_back(new RowVector(neuronLayers.back()->size()));
        deltas.push_back(new RowVector(neuronLayers.back()->size()));
  
        // vector.back() gives the handle to recently added element 
        // coeffRef gives the reference of value at that place  
        // (using this as we are using pointers here) 
        if (i != topology.size() - 1) {
            // set bias neuron (always 1, because?? weight is the bias)
            neuronLayers.back()->coeffRef(topology[i]) = 1.0; 
            cacheLayers.back()->coeffRef(topology[i]) = 1.0; 
        } 
  
        // initialze weights matrix 
        if (i > 0) { // weights matrix start from first hidden layer
            if (i != topology.size() - 1) {
                // since number of columns in this weights matrix
                // is same as the number of neurons in current layer,
                // and for all hidden and input layer, we have a bias neuron,
                // we add one to size
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1)); 
                weights.back()->setRandom(); 
                weights.back()->col(topology[i]).setZero(); 
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0; 
            } 
            else {
                // output layer does not have bias neuron
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i])); 
                weights.back()->setRandom(); 
            } 
        } 
    } 
} 

void NeuralNetwork::propagateForward(const RowVector& input) {
    // set the input to input layer and leave the bias neuron as it is
    // block returns a part of the given vector or matrix
    // block takes 4 arguments : startRow, startCol, blockRows, blockCols
    cacheLayers.front()->block(0, 0, 1, cacheLayers.front()->size() - 1) = input;

    // propagate the data forawrd
    for (uint i = 1; i < topology.size(); i++) {
        // already explained above
        (*cacheLayers[i]) = (*cacheLayers[i - 1]) * (*weights[i - 1]);
    }

    // apply the activation function to your network
    // unaryExpr applies the given function to all elements of CURRENT_LAYER
    for (uint i = 1; i < topology.size() - 1; i++) {
        for(uint c = 0; c < topology[i]; c++) {
            neuronLayers[i]->coeffRef(c) = activationFunction(cacheLayers[i]->coeffRef(c));
        }
    }

    for(uint c = 0; c < topology.back(); c++) {
        neuronLayers.back()->coeffRef(c) = cacheLayers.back()->coeffRef(c);
    }
}

void NeuralNetwork::calcErrors(const RowVector& output) {
    // calculate the errors made by neurons of last layer 
    (*deltas.back()) = output - (*neuronLayers.back()); 

    // error calculation of hidden layers is different
    // we will begin by the last hidden layer 
    // and we will continue till the first hidden layer 
    for (uint i = topology.size() - 2; i > 0; i--) { 
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose()); 
    } 
}

void NeuralNetwork::updateWeights()  {
    // topology.size()-1 = weights.size() 
    for (uint i = 0; i < topology.size() - 1; i++) { 
        // in this loop we are iterating over the different layers (from first hidden to output layer) 
        // if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols 
        // if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1 
        if (i != topology.size() - 2) { 
            for (uint c = 0; c < weights[i]->cols() - 1; c++) { 
                for (uint r = 0; r < weights[i]->rows(); r++) { 
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                } 
            } 
        } 
        else { 
            for (uint c = 0; c < weights[i]->cols(); c++) { 
                for (uint r = 0; r < weights[i]->rows(); r++) { 
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r); 
                } 
            } 
        } 
    } 
} 

void NeuralNetwork::propagateBackward(const RowVector& output)
{ 
    calcErrors(output); 
    updateWeights(); 
}

void NeuralNetwork::train(const Data& input_data, const Data& output_data) {
    for (uint i = 0; i < input_data.size(); i++) { 
        std::cout << "Input to neural network is : " << *input_data[i] << std::endl; 
        propagateForward(*input_data[i]); 
        std::cout << "Expected output is : " << *output_data[i] << std::endl; 
        std::cout << "Output produced is : " << *neuronLayers.back() << std::endl; 
        propagateBackward(*output_data[i]); 
        std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl; 
    } 
} 

void ReadCSV(std::string filename, std::vector<RowVector*>& data) {
    // free all memory
    for(auto p : data){
        if(p) delete p;
    }
    data.clear();

    std::ifstream file(filename); 
    std::string line, word; 

    // determine number of columns in file
    getline(file, line, '\n'); 
    std::stringstream ss(line); 
    std::vector<Scalar> parsed_vec; 

    while (getline(ss, word, ',')) {
        parsed_vec.push_back(Scalar(std::stof(word.data())));
    }
    uint cols = parsed_vec.size();
    data.push_back(new RowVector(cols));

    for (uint i = 0; i < cols; i++) {
        data.back()->coeffRef(i) = parsed_vec[i];
    } 
  
    // read the file 
    if (file.is_open()) { 
        while (getline(file, line, '\n')) { 
            std::stringstream ss(line); 
            data.push_back(new RowVector(1, cols)); 

            uint i = 0;
            while (getline(ss, word, ',')) {
                data.back()->coeffRef(i) = Scalar(std::stof(word.data()));
                i++; 
            } 
        } 
    } 
} 









