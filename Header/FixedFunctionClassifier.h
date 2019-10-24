/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */


#pragma once

#include "PixelClassifier.h"

namespace haft
{
	class FixedFunctionClassifier : public PixelClassifier
	{
	private:
		bool(*isColorValid) (float c1, float c2, float c3);
		int bgrConversion;
	public:
		FixedFunctionClassifier(
			bool(*isColorValid) (float c1, float c2, float c3), int bgrConversion) :
			isColorValid(isColorValid), bgrConversion(bgrConversion) {}

		//this classifier was edited due to strange behaviors, 
		//now it seems to work well, but this changes should be corrected

		//inline float classify(uchar y, uchar cr, uchar cb) const //TODO, put it back again
		inline float classify(uchar c1, uchar c2, uchar c3) const override
		{
			auto result = 0.f;

			if (isColorValid(c1, c2, c3))
			{
				result = 1.f;
			}

			return result;
		}

		inline int getBGRConversion() const override
		{
			return bgrConversion;
		}
	};	
}
