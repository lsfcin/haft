/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace haft
{
	class Corner
	{
	public:
	  Corner() : cos(0), dir(0) {}
	  Corner(float cos, int dir, cv::Point point) : cos(cos), dir(dir), point(point) {}
	  float cos;
	  int dir;
	  cv::Point point;
	};

	class CornersGroup : public Group
	{
	public:
	  CornersGroup(int start, int end, int max) : Group(start, end, max), direction(0.0), point(cv::Point()) {}
 
	  double direction; ///it is used to indacte the finger direction
	  cv::Point2f point; //representative point of the group
	};	
}