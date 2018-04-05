#pragma once
#include "MNISTLoader.h"

class MNISTLoaderTests
{
private:
	MNISTLoader mnistLoader;
public:
	void TestDataValueRange();
	void TestGetData();
	void TestGetTrainingData();
	void TestGetTestData();
	void RunAllTests();
};

