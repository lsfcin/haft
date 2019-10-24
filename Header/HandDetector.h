/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Util.h"
#include "UI.h"
#include "Hand.h"
#include "Corner.h"
#include "CornersExtractor.h"


namespace haft
{
	#define MIN_PERIMETER 200

	//#define USE_PAN

	class HandDetector
	{
	public:
		static void calcDetectionParameters(const unsigned int& perimeter,
			unsigned int& maxFinger,
			unsigned int& minFinger,
			unsigned int& midFinger,
			unsigned int& minGroupSize,
			unsigned int& maxGroupSize)
		{
			maxFinger = perimeter * 0.0553 + 5.7818;
			minFinger = perimeter * 0.0368 + 1.7797;
			midFinger = (maxFinger + minFinger) >> 1;
			minGroupSize = minFinger / 3;
			maxGroupSize = maxFinger * 1.5;
		}

		static void detectFingers(const std::vector<cv::Point>& contour, std::vector<cv::Point>& fingers, float minCosine = 0.0)
		{
			const unsigned int perimeter = contour.size();
			unsigned int maxFinger, minFinger, midFinger, minGroupSize, maxGroupSize, maxDistance, minGroups;
			calcDetectionParameters(perimeter, maxFinger, minFinger, midFinger, minGroupSize, maxGroupSize);

			maxDistance = minFinger * 2;
			minGroups = 1;    //min number of fingers in order to consider the group

			std::vector<unsigned int> distances;
			distances.push_back(maxFinger);
			distances.push_back(minFinger);
			distances.push_back(midFinger);

			std::vector<Corner> corners;
			CornersExtractor::extract(contour,
				distances,
				minCosine,
				minGroupSize,
				maxGroupSize,
				maxDistance,
				minGroups,
				corners);

			for (int i = 0; i < corners.size(); ++i)
				if (corners[i].dir > 0)
					fingers.push_back(corners[i].point);
		}

		static void detect(const std::vector<cv::Point>& contour, std::vector<Hand>& hands)
		{
			const unsigned int perimeter = contour.size();

			if (perimeter > MIN_PERIMETER)
			{
				unsigned int maxFinger, minFinger, midFinger, minGroupSize, maxGroupSize, maxDistance, minHandSize, minGroups;
				calcDetectionParameters(perimeter, maxFinger, minFinger, midFinger, minGroupSize, maxGroupSize);

				maxDistance = minFinger;
				minHandSize = 3; //min number of open fingers in order to detect hand
				minGroups = (minHandSize << 1) - 1; //indicates that must be at least 2*minHandSize-1 valleys and hills

				const float minCosine = 0.0;

				std::vector<unsigned int> distances;
				distances.push_back(maxFinger);
				distances.push_back(minFinger);
				distances.push_back(midFinger);

				std::vector<Corner> corners;
				CornersExtractor::extract(contour,
					distances,
					minCosine,
					minGroupSize,
					maxGroupSize,
					maxDistance,
					minGroups,
					corners);

				//detecting hand
				int currDirection = 1;
				int lastDirection = -1;
				cv::Point point;
				std::vector<cv::Point> fingers;
				std::vector<cv::Point> junctions;

				for (int i = 0; i < corners.size(); ++i)
				{
					point = corners[i].point;
					currDirection = corners[i].dir;

					//if the direction of the last and the current fingers midpoints are oposite
					if (currDirection * lastDirection < 0)
					{
						if (currDirection > 0)
							fingers.push_back(point);
						else
							junctions.push_back(point);

						lastDirection = corners[i].dir;
					}
					//the last case if either the first or a consecutive point is wrong directed
					else
					{
						detectHand(fingers, junctions, minHandSize, hands);

						lastDirection = -1;
						fingers.clear();
						junctions.clear();
					}
				}
				detectHand(fingers, junctions, minHandSize, hands);
			}
		}
		static void detect(const std::vector<std::vector<cv::Point>>& contours, std::vector<Hand>& hands)
		{
			const unsigned numOfContours = contours.size();

			for (int i = 0; i < numOfContours; ++i)
			{
				std::vector<cv::Point> contour = contours[i];
				detect(contour, hands);
			}
		}

	private:
		static bool patternDetected(std::vector<cv::Point>& fingers,
			std::vector<cv::Point>& junctions,
			int offset)
		{
			const float diagonalFactor = 2.0;
			const float dstMaxDiff = 1.4;

			bool result = false;

			float d11 = Util::distanceEuclidean(fingers[offset], fingers[offset + 1]); //distances between finger points
			float d12 = Util::distanceEuclidean(fingers[offset + 1], fingers[offset + 2]);
			float d2 = Util::distanceEuclidean(junctions[offset], junctions[offset + 1]); //distance between junctions
			float d31 = Util::distanceEuclidean(fingers[offset], junctions[offset]); //distances between finger and junctions
			float d32 = Util::distanceEuclidean(fingers[offset + 1], junctions[offset]);
			float d33 = Util::distanceEuclidean(fingers[offset + 1], junctions[offset + 1]);
			float d34 = Util::distanceEuclidean(fingers[offset + 2], junctions[offset + 1]);

			d31 *= diagonalFactor;
			d32 *= diagonalFactor;
			d33 *= diagonalFactor;
			d34 *= diagonalFactor;

			if (d2 < d11 && d2 < d12 && //........................//distance between junctions must be lower than the one between fingers
				d11 < d12 * dstMaxDiff && d12 < d11 * dstMaxDiff && //both distances between fingers must not differ much
				d11 < d31 && d11 < d32 && d11 < d33 && d11 < d34 && //distances between fingers must be lower than the diagonals (finger-junction)
				d12 < d31 && d12 < d32 && d12 < d33 && d12 < d34) ////same as above   
			{
				result = true;
			}

			return result;
		}
		static void detectHand(std::vector<cv::Point>& fingers,
			std::vector<cv::Point>& junctions,
			int minHandSize,
			std::vector<Hand>& hands)
		{
			int minPointsSize = (minHandSize << 1) - 1;

			int i;
			for (i = 0; i < fingers.size(); ++i)
			{
				//detected!
				if (fingers.size() - i >= minHandSize)
				{
					if (patternDetected(fingers, junctions, i))
					{
						hands.push_back(Hand(fingers, junctions, i));
						i += minHandSize; //maybe there is another hand in the same contour, this case a shift is done in the search
					}
				}
			}
		}
	};
}

//rendering mid points
//if(fingersMap[id2].info > 0)
//{
//  UI::showCircle(fingersMap[id2].point, CV_RGB(20, fingersMap[id2].info * 255, 20), 5, 2);
//}
//else
//{
//  UI::showCircle(fingersMap[id2].point, CV_RGB(fingersMap[id2].info * 255 * -1, 20, 20), 5, 2);
//}

//rendering all corners
//for(int k = fingersMap[id2].start; k < fingersMap[id2].start + fingersMap[id2].size(); ++k)
//{
//  id3 = k % corners.size();
//
//  if(fingersMap[id2].direction > 0)
//  {
//    UI::showCircle(corners[id3].point, CV_RGB(0, fingersMap[id2].direction * 255, 0));
//  }
//  else
//  {
//    UI::showCircle(corners[id3].point, CV_RGB(fingersMap[id2].direction * 255 * -1, 0, 0));
//  }
//}


//rendering
//for(int i = 0; i < handsMap.size(); ++i)
//{
//  id1 = i;
//  for(int j = handsMap[id1].start; j < handsMap[id1].start + handsMap[id1].size(); ++j)
//  {
//    id2 = j % fingersMap.size();
//    for(int k = fingersMap[id2].start; k < fingersMap[id2].start + fingersMap[id2].size(); ++k)
//    {
//      id3 = k % corners.size();
//      if(fingersMap[id2].info > 0)
//        UI::showCircle(corners[id3].point, GREEN);
//      else
//        UI::showCircle(corners[id3].point, RED);
//    }
//  }
//}

//int stop = 1;
//debug rendering code
//for(int i = 0; i < handsMap.size(); ++i)
//{
//  id1 = i;
//  for(int j = handsMap[id1].start; j < handsMap[id1].start + handsMap[id1].size(); ++j)
//  {
//    id2 = j % fingersMap.size();
//    for(int k = fingersMap[id2].start; k < fingersMap[id2].start + fingersMap[id2].size(); ++k)
//    {
//      id3 = k % corners.size();
//      UI::showCircle(corners[id3].point, WHITE);
//    }
//  }
//}