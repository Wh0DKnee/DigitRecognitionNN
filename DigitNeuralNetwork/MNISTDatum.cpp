#include "MNISTDatum.h"
#include <iostream>

using namespace std;

MNISTDatum::MNISTDatum()
{
	label = -1;
}

void MNISTDatum::Print()
{
	for (size_t i = 0; i < 28; i++)
	{
		for (size_t j = 0; j < 28; j++)
		{
			char c = '0';
			if (data(i + j*i) != 0)
			{
				c = ' ';
			}
			cout << c;
		}
		cout << endl;
	}
	cout << endl;
	cout << label << endl;
}

Eigen::VectorXf MNISTDatum::GetData()
{
	return data;
}

int MNISTDatum::GetLabel()
{
	return label;
}
