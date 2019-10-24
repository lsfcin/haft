/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "PixelClassifier.h"

namespace haft
{

	class YCrCbHSVClassifier : public PixelClassifier
	{
	public:
		const int Y_MIN = 0;
		const int Y_MAX = 255;
		const int Cr_MIN = 133;
		const int Cr_MAX = 173;
		const int Cb_MIN = 77;
		const int Cb_MAX = 127;

		YCrCbHSVClassifier() {}

		inline float classify(uchar b, uchar g, uchar r) const override
		{
			cv::Mat pixelBGR(1, 1, CV_8UC3);
			cv::Mat pixelHSV(1, 1, CV_8UC3), pixelYCbCr;

			pixelBGR.at<cv::Vec3b>(0, 0)[0] = b;
			pixelBGR.at<cv::Vec3b>(0, 0)[1] = g;
			pixelBGR.at<cv::Vec3b>(0, 0)[2] = r;

			cv::cvtColor(pixelBGR, pixelHSV, CV_BGR2HSV);
			cv::cvtColor(pixelBGR, pixelYCbCr, CV_BGR2YCrCb);

			cv::inRange(pixelYCbCr, cv::Scalar(Y_MIN, Cr_MIN, Cb_MIN), cv::Scalar(Y_MAX, Cr_MAX, Cb_MAX), pixelYCbCr);

			uchar h = pixelHSV.at<cv::Vec3b>(0, 0)[0];

			if (h > 0 && h < 17 && pixelHSV.at<uchar>(0, 0) != ' ') return 1.0f;

			return 0.0f;
		}

		inline int getBGRConversion() const override
		{
			return CV_BGR;
		}
	};
}
