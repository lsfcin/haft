/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>
#include <string.h>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "Segmenter.h"
#include "ContourExtractor.h"

#include "UI.h"

namespace haft
{
#define RADIAN_TO_DEGREES 57.29577951307855

	class QuadsFinder
	{
	public:
		static bool isQuad(std::vector<cv::Point> points, int d = 10)
		{
			//a idéia é achar quatro ângulos que somem algo perto de 360	
			//cada inclinação é a derivada dos pontos a uma distância d
			std::vector<cv::Point2f> directions(points.size());

			//a segunda derivada representa as quinas, 
			//justamente a diferença de inclinação
			std::vector<int> angles(points.size());

			//calculating all directions
			//declarations
			int iNext, iPrev;
			float norm;
			for (int i = 0; i < points.size(); ++i)
			{
				iNext = i + d / 2;
				iPrev = i - d / 2;

				if (iNext > points.size() - 1) iNext = iNext - points.size();
				if (iPrev < 0) iPrev = points.size() + iPrev;

				directions[i] = (points[iNext] - points[iPrev]);
				//normalize directions to further calc cosines
				norm = sqrt(pow((double)(directions[i].x), 2) + pow((double)(directions[i].y), 2));
				directions[i].x /= norm;
				directions[i].y /= norm;
			}

			//calculating all cosines
			for (int i = 0; i < directions.size(); ++i)
			{
				iNext = i + d / 2;
				iPrev = i - d / 2;

				if (iNext > points.size() - 1) iNext = iNext - points.size();
				if (iPrev < 0) iPrev = points.size() + iPrev;

				float cosine = directions[iNext].ddot(directions[iPrev]);
				angles[i] = std::acos(cosine) * RADIAN_TO_DEGREES;
			}

			//selecting corners	
			std::vector<int> cornersIDs;
			int maxAngle = 0;
			int maxAngleID = -1;
			int minAngleThreshold = 20;
			int minConsecutiveEdgePoints = 20;
			int consecutiveEdgePoints = 15; //starts at a high count to help corners in the beginning

			for (int i = 0; i < angles.size(); ++i)
			{
				//selecting max angle in a row, but only if there are enough edge points behind it
				if (angles[i] > minAngleThreshold && consecutiveEdgePoints > minConsecutiveEdgePoints)
				{
					//if the current angle is the greatest in this row, until now
					if (angles[i] > maxAngle)
					{
						//then update the search data
						maxAngle = angles[i];
						maxAngleID = i;
					}
				}
				else
				{
					//in case we're coming from a row of good angles
					if (maxAngle > 0)
					{
						//then, the maxAngleID is the point with greatest valuein last row, add it as a corner
						cornersIDs.push_back(maxAngleID);

						//then reset all data for the row search
						maxAngle = 0;
						maxAngleID = -1;
						consecutiveEdgePoints = 0;
					}
					else
					{
						//another edge point
						++consecutiveEdgePoints;
					}
				}
			}

			//TODO
			//solve cycling problems


			bool result = false;
			if (cornersIDs.size() == 4)
				result = true;

			return result;
		}

		static void findQuads(const cv::Mat& input, cv::Mat& output)
		{
			debugImage = input;

			output = cv::Mat::zeros(cv::Size(input.cols, input.rows), 0);

			cv::Mat gray; // = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
			cv::Mat edges;// = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
			cv::cvtColor(input, gray, CV_BGR2GRAY);

			static int c1 = 40; static int c2 = 70; static int minSize = 20;
			cv::Canny(gray, edges, c1, c2, 3, true);

			std::vector<std::vector<cv::Point>> contours;
			ContourExtractor::extract(edges, contours, 50, 50, 1);

			for (int i = 0; i < contours.size(); ++i)
			{
				if (isQuad(contours[i]))
				{
					UI::showCircles(contours[i]);
					UI::showCircles(contours[i], cv::Scalar(255, 255, 255, 0), 1, -1, output);
				}
			}

			cv::imwrite("finalResult.bmp", input);
			cv::imwrite("obtainedQuads.bmp", output);
		}

	};
}