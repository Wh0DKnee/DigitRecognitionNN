#pragma once
#include <vector>
#include <Eigen\core>
#include "GaussianRNG.h"
#include "MNISTDatum.h"

// Neural Network for classifying hand-written digits.
// Most code is based on http://neuralnetworksanddeeplearning.com/chap1.html

class Network
{
private:
	// Including input and output layers
	unsigned int layerCount;
	// Number of neurons in each layer
	std::vector<int> sizes;
	std::vector<Eigen::MatrixXf> weights;
	std::vector<Eigen::VectorXf> biases;

	// Random number generator for initializing weights and biases
	GaussianRNG gaussianRNG;
	void InitializeBiasesAndWeights();
	void PopulateNetworkRandomly();
	void UpdateMiniBatch(const std::vector<MNISTDatum>& miniBatch, float eta);
	void Backprop(std::vector<Eigen::VectorXf> &deltaNablaB, std::vector<Eigen::MatrixXf> &deltaNablaW, const MNISTDatum& datum);
	Eigen::VectorXf CostDerivative(const Eigen::VectorXf& outputActivations, const Eigen::VectorXf& y);

	// Input x has to be between 0 and 9, including both.
	// Given input 5, this will return (0, 0, 0, 0, 0, 1, 0, 0, 0, 0) transposed,
	// used to compare to network output vector
	Eigen::VectorXf CreateTruthVectorFromNumber(int number);

	static float Sigmoid(float z);
	static float SigmoidPrime(float z);

public:
	Network(std::vector<int> sizes);
	// Construct Network with specific biases and weights
	Network(std::vector<int> sizes, std::vector<Eigen::VectorXf> biases, std::vector<Eigen::MatrixXf> weights);

	// Returns the output vector corresponding to the inputs.
	Eigen::VectorXf FeedForward(Eigen::VectorXf inputs);
	// Stochastic gradient descent
	void SGD(std::vector<MNISTDatum> MNISTData, int epochs, int miniBatchSize, float eta, const std::vector<MNISTDatum>& testData);
	int Evaluate(const std::vector<MNISTDatum>& testData);
	const std::vector<Eigen::MatrixXf>& GetWeights();
	const std::vector<Eigen::VectorXf>& GetBiases();
	void PrintWeights();
	void PrintBiases();
};

