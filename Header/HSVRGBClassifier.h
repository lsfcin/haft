/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "PixelClassifier.h"
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace haft
{
	class HSVRGBClassifier : public PixelClassifier
	{
		public:
			HSVRGBClassifier() {}
		
			inline float classify (uchar b, uchar g, uchar r) const
			{
				cv::Mat pixelRGB(1, 1, CV_8UC3);
				cv::Mat pixelHSV(1, 1, CV_8UC3);

				pixelRGB.at<cv::Vec3b>(0, 0)[0] = b;
				pixelRGB.at<cv::Vec3b>(0, 0)[1] = g;
				pixelRGB.at<cv::Vec3b>(0, 0)[2] = r;

				cv::cvtColor(pixelRGB, pixelHSV, CV_BGR2HSV);

				uchar h = (int)pixelHSV.at<cv::Vec3b>(0, 0)[0],
					  s = (int)pixelHSV.at<cv::Vec3b>(0, 0)[1],
					  v = (int)pixelHSV.at<cv::Vec3b>(0, 0)[2];

				if (
					(((r > 220) && (g < 210) && (b < 170) && abs(r - g) > 15 && (r > g) && (g > b)) || ((r > 95) && (g > 40) && (b > 20) && ((std::max({r, g, b}) - std::min({r, g, b})) > 15))) 
					&& 
					((h >= 0) && (h <= 80) && (s > 70) && (s < 250) && (v > 10))
					) 
					return 1.0f;
				
				else return 0.0f;
			}

			inline int getBGRConversion() const {return CV_BGR;}

			inline bool readyToSegment() const { return true; }
	};	
}