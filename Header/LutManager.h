/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "PixelClassifier.h"

namespace haft
{
	class LUTManager
	{
	public:
		static void saveLUT(const PixelClassifier& classifier, const std::string& lutURL, float closure, const unsigned int& quantization = 256)
		{
			std::ofstream lutFile(lutURL);

			if (!lutFile)
			{
				std::cout << "Error on oppening the file" << std::endl;
				exit(1);
			}

			lutFile << quantization << std::endl;

			cv::Mat originalPixel(1, 1, CV_8UC3);
			cv::Mat convertedPixel(1, 1, CV_8UC3);
			cv::Mat segmentedPixel(1, 1, CV_8UC3);

			int index = 0;
			int increment = 256 / quantization;

			for (int r = 0; r < 256; r += increment)
			{
				for (int g = 0; g < 256; g += increment)
				{
					for (int b = 0; b < 256; b += increment)
					{
						originalPixel.data[0] = b;
						originalPixel.data[1] = g;
						originalPixel.data[2] = r;

						int conversion = classifier.getBGRConversion();

						if (conversion >= 0) cv::cvtColor(originalPixel, convertedPixel, classifier.getBGRConversion());
						else convertedPixel = originalPixel.clone();

						float classification = classifier.classify(convertedPixel.data[0], convertedPixel.data[1], convertedPixel.data[2]);

						/* if (classification >= closure)
							 classification = 1.0f;
							 else
							 classification = 0.0f;*/

						unsigned int valueInt = classification * 255;
						uchar value = (uchar)valueInt;

						lutFile << value << " ";

						//lutFile << std::hex << int(value) << int(' '); // its puts on file in hexadecimal
					}
				}
			}

			lutFile.close();
		}

		static void loadLUT(const std::string& lutURL, std::vector<uchar>& lut)
		{
			std::ifstream lutFile(lutURL);

			if (!lutFile)
			{
				std::cout << "Error on oppening the file" << std::endl;
				exit(1);
			}

			unsigned int quantization;
			lutFile >> quantization;

			const unsigned int lutSize = pow(quantization, 3);
			lut.resize(lutSize);

			for (unsigned int i = 0; i < lutSize; ++i)
			{
				lutFile >> std::hex >> lut[i];
			}

			lutFile.close();
		}

	private:
		LUTManager(){}
		~LUTManager() {}
		LUTManager(LUTManager const&);            //hiding the copy constructor
		LUTManager& operator=(LUTManager const&); //hiding the assignment operator
	};	
}