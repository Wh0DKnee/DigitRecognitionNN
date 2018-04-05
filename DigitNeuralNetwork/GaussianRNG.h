#pragma once
#include <random>

class GaussianRNG {
private:
	float mean;
	float standardDeviation;
	std::random_device rd;
	std::mt19937 e2;
	std::normal_distribution<> dist;

public:
	GaussianRNG(float mean = 0, float standardDeviation = 1) : mean(mean), standardDeviation(standardDeviation)
	{
		e2 = std::mt19937(rd());
		dist = std::normal_distribution<>(mean, standardDeviation);
	}

	float random() 
	{
		return dist(e2);
	}
};
