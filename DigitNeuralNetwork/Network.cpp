#include "Network.h"
#include <random>
#include "GaussianRNG.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include "ProgressBar.h"

using namespace std;
using namespace Eigen;

Network::Network(vector<int> sizes) : sizes(sizes)
{
	layerCount = sizes.size();
	InitializeBiasesAndWeights();
	PopulateNetworkRandomly();
}

void Network::InitializeBiasesAndWeights()
{
	for (size_t i = 0; i < layerCount - 1; i++)
	{
		weights.push_back(MatrixXf(sizes[i + 1], sizes[i]));
		biases.push_back(VectorXf(sizes[i + 1]));
	}
}

void Network::PopulateNetworkRandomly()
{
	for (size_t i = 0; i < weights.size(); i++)
	{
		for (size_t j = 0; j < weights[i].rows(); j++)
		{
			for (size_t k = 0; k < weights[i].cols(); k++)
			{
				weights[i](j, k) = gaussianRNG.random();
			}
		}
	}
	for (VectorXf &bias : biases)
	{
		for (size_t j = 0; j < bias.size(); j++)
		{
			bias(j) = gaussianRNG.random();
		}
	}
}

Network::Network(vector<int> sizes, vector<VectorXf> biases, vector<MatrixXf> weights) : sizes(sizes)
{
	layerCount = sizes.size();
	if (biases.size() != layerCount - 1 || weights.size() != layerCount - 1)
	{
		throw invalid_argument("the weights or biases are not of length layercount - 1");
	}
	for (size_t i = 0; i < layerCount - 1; i++)
	{
		if (weights[i].cols() != sizes[i] || weights[i].rows() != sizes[i + 1] || biases[i].rows() !=
			sizes[i + 1] || biases[i].cols() != 1)
		{
			throw invalid_argument("the weights or biases don't have the correct dimensions for a network of your input dimensions.");
		}
	}
	this->biases = biases;
	this->weights = weights;
}

VectorXf Network::FeedForward(VectorXf inputs)
{
	if (inputs.size() != sizes[0]) {
		throw invalid_argument("Your input vector is of different size than the input layer of the network.");
	}
	for (size_t layer = 0; layer < layerCount - 1; layer++)
	{
		inputs = ((weights[layer] * (inputs)) + biases[layer]).unaryExpr(&Network::Sigmoid);
	}
	return inputs;
}

// Stochastic gradient descent
void Network::SGD(vector<MNISTDatum> MNISTData, int epochs, int miniBatchSize, float eta, vector<MNISTDatum> testData)
{
	int dataCount = MNISTData.size();
	cout << "training data size " << dataCount << endl;
	int testDataCount = testData.size();
	int miniBatchCount = dataCount / miniBatchSize;
	cout << "miniBatchCount: " << miniBatchCount << endl;
	cout << "miniBatches size: " << miniBatchSize << endl;

	for (size_t i = 0; i < epochs; i++)
	{
		auto rng = default_random_engine{};
		rng.seed(time(0));
		shuffle(MNISTData.begin(), MNISTData.end(), rng);
		
		vector<MNISTDatum>::const_iterator first = MNISTData.begin();
		vector<MNISTDatum>::const_iterator last = MNISTData.begin() + miniBatchSize;
		vector<vector<MNISTDatum>> miniBatches;
		for (size_t j = 0; j < miniBatchCount; j++)
		{
			miniBatches.push_back(vector<MNISTDatum>(first + (j*miniBatchSize), last + (j*miniBatchSize)));
		}

		int mB = 0;
		for (vector<MNISTDatum> miniBatch : miniBatches)
		{
			if (mB % 100 == 0) // Update progress bar every 100 mini batches
			{
				ProgressBar::DrawProgressBar((float)mB / (float)miniBatchCount);
			}
			UpdateMiniBatch(miniBatch, eta);
			mB++;
		}
		cout << endl;
		if (testDataCount > 0)
		{
			cout << "Epoch: " << i << ", " << Evaluate(testData) << " correct out of " << testDataCount << endl;
		}
		else
		{
			cout << "Epoch " << i << " complete." << endl;
		}
		system("pause");
	}
	return;
}

int Network::Evaluate(std::vector<MNISTDatum> testData)
{	
	int correctCount = 0;
	int maxIndex = 0;
	for (MNISTDatum testDatum : testData)
	{
		VectorXf output = FeedForward(testDatum.GetData());
		float maxVal = output.maxCoeff();
		for (size_t i = 0; i < output.size(); i++)
		{
			if (output[i] == maxVal)
			{
				maxIndex = i;
			}
		}
		if (maxIndex == testDatum.GetLabel())
		{
			correctCount++;
		}
	}
	return correctCount;
}

void Network::UpdateMiniBatch(vector<MNISTDatum> miniBatch, float eta)
{
	vector<VectorXf> nablaB;
	vector<MatrixXf> nablaW;
	for (size_t i = 0; i < layerCount - 1; i++)
	{
		// initialize nablaB and nablaW to the proper size
		nablaB.push_back(VectorXf::Constant(sizes[i + 1], 0.f));
		nablaW.push_back(MatrixXf::Constant(sizes[i + 1], sizes[i], 0.f));
	}
	vector<VectorXf> deltaNablaB(nablaB);
	vector<MatrixXf> deltaNablaW(nablaW);

	for (MNISTDatum datum : miniBatch)
	{
		Backprop(deltaNablaB, deltaNablaW, datum);
		for (size_t i = 0; i < layerCount - 1; i++)
		{
			nablaB[i] = nablaB[i] + deltaNablaB[i];
			nablaW[i] = nablaW[i] + deltaNablaW[i];
		}
	}
	for (size_t i = 0; i < layerCount - 1; i++)
	{
		biases[i] = biases[i] - (eta / miniBatch.size())*nablaB[i];
		weights[i] = weights[i] - (eta / miniBatch.size())*nablaW[i];
	}
}

void Network::Backprop(vector<VectorXf>& deltaNablaB, vector<MatrixXf>& deltaNablaW, MNISTDatum datum)
{
	VectorXf activation;
	vector<VectorXf> activations;
	activation = datum.GetData();
	activations.push_back(activation);
	
	vector<VectorXf> zs;
	for (size_t layer = 0; layer < layerCount - 1; layer++)
	{
		VectorXf z = weights[layer]*activation + biases[layer];
		zs.push_back(z);
		activation = z.unaryExpr(&Network::Sigmoid);
		activations.push_back(activation);
	}
	VectorXf costDerivative = CostDerivative(activations.back(), CreateTruthVectorFromNumber(datum.GetLabel()));
	VectorXf delta = costDerivative.cwiseProduct(zs.back().unaryExpr(&Network::SigmoidPrime));
	deltaNablaB.back() = delta;
	deltaNablaW.back() = delta * activations[activations.size() - 2].transpose();
	for (size_t layer = layerCount - 2; layer > 0; layer--)
	{
		VectorXf z = zs[layer - 1];
		VectorXf sp = z.unaryExpr(&Network::SigmoidPrime);
		VectorXf oldDelta(delta);
		delta.resize(sizes[layer]);
		delta = (weights[layer].transpose() * oldDelta).cwiseProduct(sp).col(0);
		deltaNablaB[layer-1] = delta;
		deltaNablaW[layer-1] = delta * activations[layer - 1].transpose();
	}
}

Eigen::VectorXf Network::CostDerivative(Eigen::VectorXf outputActivations, Eigen::VectorXf y)
{
	return (outputActivations - y);
}

Eigen::VectorXf Network::CreateTruthVectorFromNumber(int number)
{
	if (number < 0 || number > 9) 
	{
		throw invalid_argument("number has to be between 0 and 9");
	}
	VectorXf vec = VectorXf::Constant(10, 0.f);
	vec(number) = 1.f;
	return vec;
}

float Network::Sigmoid(float z)
{
	return 1.0f / (1.0f + exp(-z));
}

float Network::SigmoidPrime(float z)
{
	return Sigmoid(z)*(1 - Sigmoid(z));
}

void Network::PrintWeights()
{
	int matIndex = 0;
	for (MatrixXf mat : weights)
	{
		cout << "Weights from layer " << matIndex << " to layer " << (matIndex + 1) << ":" << endl;
		cout << mat << endl;
		matIndex++;
	}
}

std::vector<Eigen::MatrixXf> Network::GetWeights()
{
	return weights;
}

std::vector<Eigen::VectorXf> Network::GetBiases()
{
	return biases;
}

void Network::PrintBiases()
{
	int vecIndex = 0;
	for (VectorXf bias : biases)
	{
		cout << "Biases for layer " << vecIndex << ":" << endl;
		cout << bias << endl;
		vecIndex++;
	}
}
