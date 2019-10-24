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
	class Hand
{
public:
	Hand(std::vector<cv::Point> points, const int& offset = 0) //offset is the pattern detection function offset, representing the first of three fingers used in the detection
	{
		int start;
		if (offset >= 2 &&
			Util::distanceEuclidean(points[offset - 2], points[offset]) <
			Util::distanceEuclidean(points[offset + 2], points[offset]) * 1.5)
		{
			start = offset - 2;
			midFinger = 2;
		}
		else
		{
			start = offset;
			midFinger = 1;
		}

		int end;
		if (points.size() >= offset + 7 &&
			Util::distanceEuclidean(points[offset + 4], points[offset + 6]) <
			Util::distanceEuclidean(points[offset + 4], points[offset + 2]) * 1.5)
			end = offset + 7;
		else end = offset + 5;

		for (auto i = start; i < end; ++i)
		{
			if (i % 2 == 0) fingers.push_back(points[i]);
			else junctions.push_back(points[i]);
		}

		cv::Point base = Util::findPoint(junctions[midFinger - 1], junctions[midFinger], 0.5f);
		palm = Util::findPoint(fingers[midFinger], base, 1.7f);
		float radius = Util::distanceEuclidean(fingers[midFinger], palm) * 0.9;
		rect = cv::Rect(palm.x - radius, palm.y - radius, radius * 2, radius * 2);
		//UI::showLine(fingers[midFinger], palm, GREEN);
		//UI::showLine(junctions[midFinger - 1], junctions[midFinger], RED);
		//UI::showCircle(base, RED, 1);
		//UI::showCircle(palm, GREEN, radius, 1);
	}

	Hand(std::vector<cv::Point> fingers,
	     std::vector<cv::Point> junctions,
	     const int& offset = 0) //offset is the pattern detection function offset, representing the first of three fingers used in the detection
	{
		midFinger = 1;

		for (auto i = offset; i < fingers.size(); ++i)
		{
			this->fingers.push_back(fingers[i]);
			if (i < junctions.size()) this->junctions.push_back(junctions[i]);
		}

		cv::Point base = Util::findPoint(junctions[midFinger - 1], junctions[midFinger], 0.5f);
		palm = Util::findPoint(fingers[midFinger], base, 1.7f);
		float radius = Util::distanceEuclidean(fingers[midFinger], palm) * 0.9;
		rect = cv::Rect(palm.x - radius, palm.y - radius, radius * 2, radius * 2);
	}

	void render()
	{
		for (int i = 0; i < fingers.size(); ++i)
			UI::showCircle(fingers[i], GREEN, 6);

		//for(int i = 0; i < junctions.size(); ++i)
		//    UI::showCircle(junctions[i], RED, 4, 1);

		UI::showCircle(palm, GREEN, 10, 2);
		UI::showRectangle(rect, GREEN, 2);
	}

	std::vector<cv::Point> fingers;
	std::vector<cv::Point> junctions;
	cv::Point palm;
	unsigned int midFinger;
	cv::Rect rect;
};	
}

