/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "HaftDll.h"

#include "Target.h"
#include "HandDetector.h"
#include "AdaboostDetector.h"
#include "GFTTExtractor.h"

namespace haft
{
	namespace Detector
	{
		HAFTDLL_API extern bool detectHands;
		HAFTDLL_API extern bool detectFaces;
		HAFTDLL_API void detect(
			std::vector<Target>& targets,
			const std::vector<std::vector<cv::Point>>& contours,
			const cv::Mat& mask,
			const cv::Mat& currGray);
	}
}