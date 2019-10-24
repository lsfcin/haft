/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "AGFLabeler.h"
#include "UI.h"

namespace haft
{
	class Skeletonizer
	{
		static void skeletonize(cv::Mat& mask)
		{
			IplImage iplMask = mask;
			cv::Mat auxiliarMask(&iplMask, true);

			cv::Mat skel(auxiliarMask.size(), CV_8UC1, cv::Scalar(0));
			cv::threshold(auxiliarMask, auxiliarMask, 0, 255, CV_8UC1);
			cv::Mat temp;
			cv::Mat eroded;
			cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

			bool done;
			do
			{
				cv::erode(auxiliarMask, eroded, element);
				cv::dilate(eroded, temp, element); // temp = open(img)
				cv::subtract(auxiliarMask, temp, temp);
				cv::bitwise_or(skel, temp, skel);
				eroded.copyTo(auxiliarMask);
				done = (cv::norm(auxiliarMask) == 0);
			} while (!done);

			static auto minSize = 10;
			AGFLabeler::instance().label(skel);
			AGFLabeler::instance().removeSmallGroups(skel, minSize);
			cv::dilate(skel, skel, cv::Mat(), cv::Point(-1, -1), 1);
			AGFLabeler::instance().label(skel);
			AGFLabeler::instance().removeSmallGroups(skel, minSize * 4);
			AGFLabeler::instance().label(skel);
			AGFLabeler::instance().pointExtremities(skel, 4);

			UI::mapVar2Trackbar(minSize, 200, "s", "mask");
			cv::subtract(mask, skel, mask);
		}
	};
}
