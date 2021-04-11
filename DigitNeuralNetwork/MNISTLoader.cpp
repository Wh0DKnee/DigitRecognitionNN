#include "MNISTLoader.h"
#define USE_MNIST_LOADER
#define MNIST_HDR_ONLY
#include "mnist.h"
#include <stdexcept>
#include <Eigen\core>
#include <iostream>

using namespace std;
using namespace Eigen;
std::vector<MNISTDatum> MNISTLoader::GetTrainingData()
{
	return GetData(0, 50000);
}

std::vector<MNISTDatum> MNISTLoader::GetTestData()
{
	return GetData(50000, 60000);
}

std::vector<MNISTDatum> MNISTLoader::GetData(unsigned int startIndex, unsigned int endIndex)
{
	if (startIndex >= endIndex)
	{
		throw invalid_argument("startIndex must be smaller than endIndex");
	}
	if (startIndex > 60000 || endIndex > 60000)
	{
		throw invalid_argument("There are only 60000 images in the database, your index is too big.");
	}
	mnist_data *data;
	unsigned int cnt;
	int ret;

	unsigned int range = endIndex - startIndex;
	vector<MNISTDatum> MNISTData(range);

	if (ret = mnist_load("../MNISTData/train-images.idx3-ubyte", "../MNISTData/train-labels.idx1-ubyte", &data, &cnt)) {
		printf("An error occured: %d\n", ret);
	}
	else {
		float f;
		for (unsigned int x = 0; x < range; x++)
		{
			VectorXf temp(784);
			for (int i = 0; i < 28; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					f = ((*(data + x + startIndex)).data[i][j]);
					f /= 255.f;
					temp(i*28 + j) = f;
				}
			}
			MNISTData[x] = MNISTDatum(temp, (*(data + x + startIndex)).label);
		}
		free((data));
	}
	return MNISTData;
}
