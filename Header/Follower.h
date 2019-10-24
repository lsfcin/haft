/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "Target.h"
#include "PanFollower.h"

namespace haft
{
	namespace Follower
	{
		inline void follow(
			std::vector<Target>& targets,
			const cv::Mat& currGray,
			const cv::Mat& mask,
			const cv::Mat& lastGray)
		{
			if (!lastGray.empty())
			{
				for (auto i = 0; i >= 0 && i < targets.size(); ++i)
				{
					//only try to follow if there are pre-exisistent features
					const auto nLastFeatures = targets[i].lastFeatures().size();
					if (nLastFeatures > 0)
					{
						PanFollower::follow(currGray, lastGray, mask, targets[i]);
					}
				}
			}

		}
	}
}