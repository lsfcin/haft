/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "HaftDll.h"
#include "Tracker.h"
#include "Globals.h"

namespace haft
{
	HAFTDLL_API void track(cv::Mat& frame, std::vector<Target>& targets);

	HAFTDLL_API void trackFromCamera(int camID, std::vector<Target>& targets);

	HAFTDLL_API void trackFromVideo(const std::string& videoURL, std::vector<Target>& targets);

	HAFTDLL_API void trackFromImage(const std::string& imageURL, std::vector<Target>& targets);
}
