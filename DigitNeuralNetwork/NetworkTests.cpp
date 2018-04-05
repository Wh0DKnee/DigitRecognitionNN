#include "NetworkTests.h"
#include "Network.h"
#include <Eigen\core>
#include <iostream>

using namespace std;
using namespace Eigen;

void NetworkTests::TestNetworkSize()
{
	Network net(vector<int>{3, 4, 2});
	vector<MatrixXf> weights = net.GetWeights();
	vector<VectorXf> biases = net.GetBiases();
	assert(biases.size() == 2);
	assert(weights.size() == 2);
	assert(biases[0].rows() == 4);
	assert(biases[1].rows() == 2);
	assert(biases[0].cols() == 1);
	assert(biases[1].cols() == 1);
	assert(weights[0].cols() == 3);
	assert(weights[0].rows() == 4);
	assert(weights[1].cols() == 4);
	assert(weights[1].rows() == 2);
}

void NetworkTests::TestFeedForward()
{
	vector<VectorXf> biases;
	MatrixXf b1(4, 1);
	b1 << 2.f, 1.f, 2.f, 1.f;
	biases.push_back(b1);
	VectorXf b2(2);
	b2 << 1.f, -7.5f;
	biases.push_back(b2);

	vector<MatrixXf> weights;
	MatrixXf w1(4, 3);
	w1 << 0.5f, 0.5f, 0.5f,
		0.5f, 0.5f, 0.5f,
		0.5f, 0.5f, 0.5f,
		0.5f, 0.5f, 0.5f;
	weights.push_back(w1);
	MatrixXf w2(2, 4);
	w2 << 2.f, 2.f, 2.f, 2.f,
		2.f, 2.f, 2.f, 2.f;
	weights.push_back(w2);

	Network net(vector<int>{3, 4, 2}, biases, weights);
	VectorXf inputs(3);
	inputs << 1.f, 2.f, 3.f;
	float z21 = (1.f*0.5f + 2.f * 0.5f + 3.f * 0.5f) + 2.f; //5
	float z22 = (1.f*0.5f + 2.f * 0.5f + 3.f * 0.5f) + 1.f; //4
	float z23 = (1.f*0.5f + 2.f * 0.5f + 3.f * 0.5f) + 2.f; //5
	float z24 = (1.f*0.5f + 2.f * 0.5f + 3.f * 0.5f) + 1.f; //4
	float a21 = 0.9933071490757153f;
	float a22 = 0.9820137900379085f;
	float a23 = a21;
	float a24 = a22;

	float z31 = (a21 + a22 + a23 + a24) * 2.f + 1.f;
	float z32 = (a21 + a22 + a23 + a24) * 2.f + -7.5f;
	float a31 = 0.9998638040915139f;
	float a32 = 0.5989961157816596f;

	VectorXf outputs(2);
	outputs << a31, a32;
	
	assert(abs(outputs(0) - net.FeedForward(inputs)(0)) < 0.001f);
	assert(abs(outputs(1) - net.FeedForward(inputs)(1)) < 0.001f);

}
