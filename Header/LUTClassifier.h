/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#ifndef __LUTCLASSFIER_H__
#define __LUTCLASSFIER_H__

#include "PixelClassifier.h"
#include "LUTManager.h"
#include <vector>

class LUTClassifier : public PixelClassifier
{
	public :
		std::vector<uchar> lut;
		unsigned int quantization;
		unsigned int shiftL;
		unsigned int shiftR;
		unsigned int temp;

		LUTClassifier(const std::string& lutURL)
		{
			LUTManager::loadLUT(lutURL, lut);
	
			quantization = pow(lut.size(), 1.0/3.0);
			if (quantization % 2 == 1) ++quantization;
			shiftL = 0;
			shiftR = 0;
			temp = quantization;

			while (temp > 1)
			{
				temp /= 2;
				++shiftL;
			}

			temp = 256 / quantization;
			while (temp > 1)
			{
				temp /= 2;
				++shiftR;
			}
		}

		inline float classify(uchar b, uchar g, uchar r) const
		{
			unsigned int lutID = 0;
			
			lutID = (b >> shiftR) + ((g >> shiftR) << shiftL) + ((r >> shiftR) << shiftL << shiftL);
			float classification = lut[lutID];
			
			return classification/255.0;
		}

		inline int getConversion() const {return CV_BGR;}
};

#endif