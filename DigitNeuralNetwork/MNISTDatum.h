#pragma once
#include <vector>
#include <Eigen/core>

// data structure for a single MNIST image and its label
class MNISTDatum
{
private:
	Eigen::VectorXf data;
	int label;

public:
	MNISTDatum(Eigen::VectorXf data, int label) : data(data), label(label) {}
	MNISTDatum();

	void Print();
	Eigen::VectorXf GetData();
	int GetLabel();
};

