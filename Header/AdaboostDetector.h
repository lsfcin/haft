/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "HaftDll.h"

#include <opencv2/opencv.hpp>

#include "Target.h"

namespace haft
{
	class AdaboostDetector
	{
	public:
		static HAFTDLL_API AdaboostDetector& instance();
		HAFTDLL_API bool detect(const cv::Mat& grayImage, std::vector<cv::Rect>& targets, TargetType type, double scale = 1);

	private:
		cv::CascadeClassifier cascadeFace;
		cv::CascadeClassifier cascadeHand;
		cv::CascadeClassifier cascadeBody;

		AdaboostDetector();

		AdaboostDetector(AdaboostDetector const&) = delete; //hiding the copy constructor
		AdaboostDetector& operator=(AdaboostDetector const&) = delete; //hiding the assignment operator
	};
}
