/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace haft
{
	class ImageProcessor
	{
		virtual void process(const cv::Mat& input, cv::Mat& output) = 0;
	};
}