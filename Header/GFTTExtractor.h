/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "Features.h"
#include "Target.h"

namespace haft
{
	class GFTTExtractor
	{
	public:
		static GFTTExtractor& instance()
		{
			static GFTTExtractor _instance;

			return _instance;
		}

		bool extract(const cv::Mat& gray, const cv::Mat& mask, Target& target)
		{
			auto succeeded = false;

			std::vector<cv::Point2f> detectedPoints;
			succeeded = extract(gray, mask, target.roi, detectedPoints, target.maxNumFeatures);
			target.features.setPoints(detectedPoints);

			return succeeded;
		}

		bool extract(const cv::Mat& gray, const cv::Mat& mask, const cv::Rect& roi, std::vector<cv::Point2f>& points, unsigned int maxNumPoints = 30)
		{
			auto succeeded = false;

			if (roi.width > 0 && roi.height > 0)
			{
				cv::goodFeaturesToTrack(gray(roi), points, maxNumPoints, minFeatQuality, minPointsDistance, mask(roi));
				for (unsigned int i = 0; i < points.size(); ++i)
				{
					points[i].x += roi.x;
					points[i].y += roi.y;
				}
			}

			if (points.size() > minNumPoints)
			{
				if (points.size() > maxNumPoints) points.resize(maxNumPoints);
				succeeded = true;
			}
			else
			{
				succeeded = false;
			}

			return succeeded;
		}

		//visible data
		unsigned int windowSize;
		unsigned int minNumPoints;
		unsigned int minPointsDistance;
		float minFeatQuality;

	private:
		GFTTExtractor(
			unsigned int windowSize = 10,
			unsigned int minNumPoints = 10,
			unsigned int minPointsDistance = 3,
			float minFeatQuality = 0.001f) :
			windowSize(windowSize),
			minNumPoints(minNumPoints),
			minPointsDistance(minPointsDistance),
			minFeatQuality(minFeatQuality)
		{
		}
		GFTTExtractor(GFTTExtractor const&) = delete;            //hiding the copy constructor
		GFTTExtractor& operator=(GFTTExtractor const&) = delete; //hiding the assignment operator
	};
}