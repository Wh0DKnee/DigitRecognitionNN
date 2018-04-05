#include "MNISTLoaderTests.h"
#include <stdexcept>
#include <Eigen\core>

using namespace std;
using namespace Eigen;

void MNISTLoaderTests::TestDataValueRange()
{
	vector<MNISTDatum> a = mnistLoader.GetData(0, 100);
	for (size_t i = 0; i < a.size(); i++)
	{
		for (size_t p = 0; p < 784; p++)
		{
			MNISTDatum datum = a[i];
			float f = datum.GetData()[p];
			assert(f >= 0.f && f <= 1.f);
		}
	}
}

void MNISTLoaderTests::TestGetData()
{
	vector<MNISTDatum> a = mnistLoader.GetData(12, 45000);
	assert(a.size() == 45000 - 12);
}

void MNISTLoaderTests::TestGetTrainingData()
{
	vector<MNISTDatum> a = mnistLoader.GetTestData();
	assert(a.size() == 10000);
}

void MNISTLoaderTests::TestGetTestData()
{
	vector<MNISTDatum> a = mnistLoader.GetTrainingData();
	assert(a.size() == 50000);
}

void MNISTLoaderTests::RunAllTests()
{
	TestGetData();
	TestGetTestData();
	TestGetTrainingData();
	TestDataValueRange();
}
