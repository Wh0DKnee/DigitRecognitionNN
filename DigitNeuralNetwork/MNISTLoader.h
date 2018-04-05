#pragma once
#include "MNISTDatum.h"

class MNISTLoader
{
public:
	std::vector<MNISTDatum> GetTrainingData();
	std::vector<MNISTDatum> GetTestData();
	//start and end must be in range 0<=x<60000 and start<end
	std::vector<MNISTDatum> GetData(unsigned int startIndex, unsigned int endIndex);
};

