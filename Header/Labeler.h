/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "Target.h"
#include "ContourExtractor.h"

namespace haft
{
	namespace Labeler
	{
		inline void label(
			std::vector<Target>& targets,
			std::vector<std::vector<cv::Point>>& contours, 
			cv::Mat& mask,
			const cv::Mat& currGray)
		{
			//cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 5);
			//cv::GaussianBlur(mask, mask, cv::Size(9, 9), 9, 9);
			ContourExtractor::extract(mask, contours, 400, 0, CV_FILLED);

			for (unsigned int i = 0; i < targets.size(); ++i)
			{
				auto contourID = -1;

				//sort features accordingly to its distance from the median point
				auto sortedFeatures = targets[i].features;
				for (auto j = 0; j < sortedFeatures.size(); ++j)
				{
					sortedFeatures.setRelevance(1.f / Util::distance4C(targets[i].medianPoint, sortedFeatures[j].p2D), j);
				}
				if (sortedFeatures.size() > 2)
					sortedFeatures.sort();

				//then iterate and the first found contour is the one representing current target
				const auto maxFeatureID = currGray.size().area();
				for (auto j = 0; j < sortedFeatures.size(); ++j)
				{
					int featureID = Util::index1D(sortedFeatures[j].p2D, currGray.size());

					if (featureID >= 0 && featureID < maxFeatureID)
					{
						int contourIDWithOffset = mask.data[featureID];
						contourID = contourIDWithOffset - CONTOUR_VISUALIZATION_OFFSET;
						if (contourID >= 0)
							break;
					}
				}

				//update contour
				if (contourID >= 0)
				{
					Target target;

					target.maxNumFeatures = targets[i].maxNumFeatures;
					target.setType(targets[i].getType());
					target.contour = contours[contourID];
					target.contourID = contourID;
					Util::boundingBox(target.contour, mask.size(), 0.0, target.roi);
					Util::limitBox(currGray.size(), target.roi);

					targets[i].update(target);
					targets[i].succeeded = true;
				}


				else
				{
					//try to recover the target
					auto recovered = false;

					if (!recovered) //else, target is lost 
					{
						targets.erase(targets.begin() + i);
						--i;
					}
				}
			}
		}
	}
}