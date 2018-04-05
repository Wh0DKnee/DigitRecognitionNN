#define USE_MNIST_LOADER
#include "mnist.h"
#include "Network.h"
#include <iostream>
#include "MNISTDatum.h"
#include "MNISTLoader.h"
#include "MNISTLoaderTests.h"
#include "NetworkTests.h"
#include "Eigen\core"

using namespace std;
using namespace Eigen;

void MakeAndTrainNeuralNet() 
{
	MNISTLoader loader;
	Network net(std::vector<int>{784, 30, 10});
	//net.PrintWeights();
	//net.PrintBiases();
	cout << endl << endl << endl;
	vector<MNISTDatum> trainingData = loader.GetTrainingData();
	vector<MNISTDatum> testData = loader.GetTestData();
	net.SGD(trainingData, 30, 10, 3.0f, testData);
}

int main()
{
	/*NetworkTests netTests;
	netTests.TestNetworkSize();
	netTests.TestFeedForward();*/
	MakeAndTrainNeuralNet();

	return 0;
}