#include "ProgressBar.h"
#include <iostream>

namespace ProgressBar
{
	void DrawProgressBar(float percentage)
	{
		float progress = percentage;
		
		int barWidth = 70;

		std::cout << "[";
		int pos = barWidth * progress;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << int(progress * 100.0) << " %\r";
	}
}
