/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "PixelClassifier.h"

namespace haft
{
	class YCrCbGarciaTziritas : public PixelClassifier
	{
	public:
		YCrCbGarciaTziritas() {}

		//this classifier was edited due to strange behaviors, 
		//now it seems to work well, but this changes should be corrected

		//inline float classify(uchar y, uchar cr, uchar cb) const //TODO, put it back again
		inline float classify(uchar y, uchar cb, uchar cr) const override
		{
			double theta1;
			double theta2;
			double theta3;
			double theta4;

			if (y > 128)
			{
				theta1 = -2 + ((256 - y) / 16);
				theta2 = 20 - ((356 - y) / 16);
				theta3 = 6;
				theta4 = -8;
			}
			else
			{
				theta1 = 6;
				theta2 = 12;
				theta3 = 2 + (y / 32);
				theta4 = -16 + (y / 16);
			}

			cr -= 128;
			cb -= 128;

			bool c1 = cr >= -2 * (cb + 24); bool c2 = cr >= -(cb + 17);
			bool c3 = cr >= -4 * (cb + 32); bool c4 = cr >= 2.5  * (cb + theta1);
			bool c5 = cr >= theta3;        bool c6 = cr >= 0.5  * (theta4 - cb);
			bool c7 = cr <= (220 - cb) / 6; bool c8 = cr <= (4 / 3) * (theta2 - cb);

			float result;
			//if(c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8) //TODO, put it back again
			if (c1 && c2 && c3 && c4 && c5 && c6)
				result = 1.0f;
			else
				result = 0.0f;

			return result;
		}
		inline int getBGRConversion() const override { return CV_BGR2YCrCb; }
		inline bool readyToSegment() const { return true; }
	};
}