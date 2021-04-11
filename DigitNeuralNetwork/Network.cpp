#include "Network.h"
#include <random>
#include "GaussianRNG.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include "ProgressBar.h"

using namespace Eigen;

Network::Network(std::vector<int> sizes) : sizes(sizes)
{
	layerCount = sizes.size();
	InitializeBiasesAndWeights();
	PopulateNetworkRandomly();
}

void Network::InitializeBiasesAndWeights()
{
	for (unsigned int i = 0; i < layerCount - 1; i++)
	{
		weights.push_back(MatrixXf(sizes[i + 1], sizes[i]));
		biases.push_back(VectorXf(sizes[i + 1]));
	}
}

void Network::PopulateNetworkRandomly()
{
	for (size_t i = 0; i < weights.size(); i++)
	{
		for (Index j = 0; j < weights[i].rows(); j++)
		{
			for (Index k = 0; k < weights[i].cols(); k++)
			{
				weights[i](j, k) = gaussianRNG.random();
			}
		}
	}
	for (VectorXf &bias : biases)
	{
		for (Eigen::Index j = 0; j < bias.size(); j++)
		{
			bias(j) = gaussianRNG.random();
		}
	}
}

Network::Network(std::vector<int> sizes, std::vector<VectorXf> biases, std::vector<MatrixXf> weights) : sizes(sizes)
{
	layerCount = sizes.size();
	if (biases.size() != layerCount - 1 || weights.size() != layerCount - 1)
	{
		throw std::invalid_argument("the weights or biases are not of length layercount - 1");
	}
	for (unsigned int i = 0; i < layerCount - 1; i++)
	{
		if (weights[i].cols() != sizes[i] || weights[i].rows() != sizes[i + 1] || biases[i].rows() !=
			sizes[i + 1] || biases[i].cols() != 1)
		{
			throw std::invalid_argument("the weights or biases don't have the correct dimensions for a network of your input dimensions.");
		}
	}
	this->biases = biases;
	this->weights = weights;
}

VectorXf Network::FeedForward(VectorXf inputs)
{
	if (inputs.size() != sizes[0]) {
		throw std::invalid_argument("Your input vector is of different size than the input layer of the network.");
	}
	for (unsigned int layer = 0; layer < layerCount - 1; layer++)
	{
		inputs = ((weights[layer] * (inputs)) + biases[layer]).unaryExpr(&Network::Sigmoid);
	}
	return inputs;
}

// Stochastic gradient descent
void Network::SGD(std::vector<MNISTDatum> MNISTData, int epochs, int miniBatchSize, float eta, const std::vector<MNISTDatum>& testData)
{
	int dataCount = MNISTData.size();
	std::cout << "training data size " << dataCount << std::endl;
	int testDataCount = testData.size();
	int miniBatchCount = dataCount / miniBatchSize;
	std::cout << "miniBatchCount: " << miniBatchCount << std::endl;
	std::cout << "miniBatches size: " << miniBatchSize << std::endl;

	for (int i = 0; i < epochs; i++)
	{
		auto rng = std::default_random_engine{};
		rng.seed(static_cast<unsigned int>(time(0)));
		shuffle(MNISTData.begin(), MNISTData.end(), rng);
		
		std::vector<MNISTDatum>::const_iterator first = MNISTData.begin();
		std::vector<MNISTDatum>::const_iterator last = MNISTData.begin() + miniBatchSize;
		std::vector<std::vector<MNISTDatum>> miniBatches; // TODO: implement miniBatches with indexes to avoid copying data
		for (int j = 0; j < miniBatchCount; j++)
		{
			miniBatches.push_back(std::vector<MNISTDatum>(first + (j*miniBatchSize), last + (j*miniBatchSize)));
		}

		int mB = 0;
		for (const std::vector<MNISTDatum>& miniBatch : miniBatches)
		{
			if (mB % 100 == 0) // Update progress bar every 100 mini batches
			{
				ProgressBar::DrawProgressBar((float)mB / (float)miniBatchCount);
			}
			UpdateMiniBatch(miniBatch, eta);
			mB++;
		}
		std::cout << std::endl;
		if (testDataCount > 0)
		{
			std::cout << "Epoch: " << i << ", " << Evaluate(testData) << " correct out of " << testDataCount << std::endl;
		}
		else
		{
			std::cout << "Epoch " << i << " complete." << std::endl;
		}
		system("pause");
	}
	return;
}

int Network::Evaluate(const std::vector<MNISTDatum>& testData)
{	
	int correctCount = 0;
	int maxIndex = 0;
	for (const MNISTDatum& testDatum : testData)
	{
		VectorXf output = FeedForward(testDatum.GetData());
		float maxVal = output.maxCoeff();
		for (Eigen::Index i = 0; i < output.size(); i++)
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

void Network::UpdateMiniBatch(const std::vector<MNISTDatum>& miniBatch, float eta)
{
	std::vector<VectorXf> nablaB;
	std::vector<MatrixXf> nablaW;
	for (unsigned int i = 0; i < layerCount - 1; i++)
	{
		// initialize nablaB and nablaW to the proper size
		nablaB.push_back(VectorXf::Constant(sizes[i + 1], 0.f));
		nablaW.push_back(MatrixXf::Constant(sizes[i + 1], sizes[i], 0.f));
	}
	std::vector<VectorXf> deltaNablaB(nablaB);
	std::vector<MatrixXf> deltaNablaW(nablaW);

	for (const MNISTDatum& datum : miniBatch)
	{
		Backprop(deltaNablaB, deltaNablaW, datum);
		for (unsigned int i = 0; i < layerCount - 1; i++)
		{
			nablaB[i] = nablaB[i] + deltaNablaB[i];
			nablaW[i] = nablaW[i] + deltaNablaW[i];
		}
	}
	for (unsigned int i = 0; i < layerCount - 1; i++)
	{
		biases[i] = biases[i] - (eta / miniBatch.size())*nablaB[i];
		weights[i] = weights[i] - (eta / miniBatch.size())*nablaW[i];
	}
}

void Network::Backprop(std::vector<VectorXf>& deltaNablaB, std::vector<MatrixXf>& deltaNablaW, const MNISTDatum& datum)
{
	VectorXf activation;
	std::vector<VectorXf> activations;
	activation = datum.GetData();
	activations.push_back(activation);
	
	std::vector<VectorXf> zs;
	for (unsigned int layer = 0; layer < layerCount - 1; layer++)
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
	for (int layer = layerCount - 2; layer > 0; layer--)
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

Eigen::VectorXf Network::CostDerivative(const Eigen::VectorXf& outputActivations, const Eigen::VectorXf& y)
{
	return (outputActivations - y);
}

Eigen::VectorXf Network::CreateTruthVectorFromNumber(int number)
{
	if (number < 0 || number > 9) 
	{
		throw std::invalid_argument("number has to be between 0 and 9");
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
	for (const MatrixXf& mat : weights)
	{
		std::cout << "Weights from layer " << matIndex << " to layer " << (matIndex + 1) << ":" << std::endl;
		std::cout << mat << std::endl;
		matIndex++;
	}
}

const std::vector<Eigen::MatrixXf>& Network::GetWeights()
{
	return weights;
}

const std::vector<Eigen::VectorXf>& Network::GetBiases()
{
	return biases;
}

void Network::PrintBiases()
{
	int vecIndex = 0;
	for (const VectorXf& bias : biases)
	{
		std::cout << "Biases for layer " << vecIndex << ":" << std::endl;
		std::cout << bias << std::endl;
		vecIndex++;
	}
}
