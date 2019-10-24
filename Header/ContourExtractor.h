/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "Util.h"
#include "UI.h"

namespace haft
{
#define CONTOUR_VISUALIZATION_OFFSET 100 //ideally should be 1 to suport 254 contours, but for debugging can be set with a higher value

	class ContourExtractor
	{
	public:
		static std::vector<std::vector<cv::Point>> extract(cv::Mat& mask, std::vector<std::vector<cv::Point>>& contours, const unsigned int& minArea = 0, const unsigned int& minPerimeter = 0, int thickness = CV_FILLED, bool polygonal = false)
		{
			cv::findContours(mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
			mask = cv::Mat::zeros(mask.size(), CV_8UC1);
			auto dilate = 0;
			if (thickness != CV_FILLED && thickness != 1)
			{
				dilate = thickness >> 1;
				thickness = 1;
			}
			for (auto i = 0; i < contours.size(); ++i)
			{
				if (cv::contourArea(contours[i]) >= minArea && contours[i].size() >= minPerimeter)
				{
					if (polygonal)
						cv::approxPolyDP(cv::Mat(contours[i]), contours[i], 3, true);

					cv::drawContours(mask, contours, i, CV_RGB(CONTOUR_VISUALIZATION_OFFSET + i,
						CONTOUR_VISUALIZATION_OFFSET + i,
						CONTOUR_VISUALIZATION_OFFSET + i), thickness);
				}
				else
				{
					contours.erase(contours.begin() + i);
					--i;
				}
			}

			if (dilate) cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), dilate);

			return contours;
		}
	};
}