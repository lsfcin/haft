/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "ContourExtractor.h"
#include "Features.h"
#include "Util.h"

namespace haft
{
#define ROI_INCREMENT 20
#define MIN_DISTANCE 10

	enum Relevance { NONE, VELOCITY, AGE };

	class PanFollower
	{
	public:
		static void nullptrifyLastPoints(cv::Mat& mask, const std::vector<cv::Point2f>& points, unsigned int radius = 5)
		{
			for (unsigned int i = 0; i < points.size(); ++i) {
				cv::circle(mask, points[i], radius, CV_RGB(0, 0, 0), CV_FILLED);
			}
		}

		static void calculateMedianPoint(Target& target,
			int factorPower = 1)
		{
			target.medianPoint = cv::Point2f(0, 0);
			double factorSum = 0;
			for (unsigned int i = 0; i < target.features.size(); ++i)
			{
				const auto factor = pow(target.features[i].relevance, factorPower);
				factorSum += factor;
				target.medianPoint.x += target.features[i].p2D.x * factor;
				target.medianPoint.y += target.features[i].p2D.y * factor;
			}
			target.medianPoint.x /= factorSum;
			target.medianPoint.y /= factorSum;
		}

		static void calculateMedianPointPan(const cv::Mat& mask,
			Target& target,
			bool useFactor = true,
			int factorPower = 2)
		{
			float distance, factor, distanceSum, bestDistanceSum = INT_MAX;
			int bestPointIndex = -1;

			for (unsigned int i = 0; i < target.features.size(); i++)
			{
				distanceSum = 0;

				//the median point can only be a valid pixel
				int featureID = Util::index1D(target.features[i].p2D, mask.size());
				int maskData = mask.data[featureID];
				if (maskData == target.contourID + CONTOUR_VISUALIZATION_OFFSET)
				{
					for (unsigned int j = 0; j < target.features.size(); j++)
					{
						if (i != j)
						{
							distance = Util::distanceEuclidean(target.features[i].p2D, target.features[j].p2D);

							if (useFactor)
							{
								factor = pow(target.features[j].relevance / (target.features[i].relevance + target.features[j].relevance + 0.00001f), (int)factorPower);
								distanceSum += (factor * distance);
							}
							else
							{
								distanceSum += distance;
							}
						}
					}

					if (distanceSum < bestDistanceSum) {
						bestPointIndex = i;
						bestDistanceSum = distanceSum;
					}
				}
			}

			if (bestPointIndex >= 0)
			{
				target.medianPoint = target.features[bestPointIndex].p2D;
			}
		}

		static void relocate(const cv::Point2f medianPoint, Features& features)
		{
			//relocating most far feature from the median point    
			int pointIndex = -1;
			float pointDistance = 0;

			for (unsigned int i = 0; i < features.size(); ++i)
			{
				float currentDistance = Util::distanceEuclidean(features[i].p2D, medianPoint);
				if (currentDistance > pointDistance)
				{
					pointDistance = currentDistance;
					pointIndex = i;
				}
			}

			if (pointIndex >= 0)
			{
				cv::Point2f p((features[pointIndex].p2D.x + medianPoint.x) / 2,
					(features[pointIndex].p2D.y + medianPoint.y) / 2);
				features.setPoint(p, pointIndex);
			}
		}

		static void removeOutliers(const cv::Rect& roi, Features& features, Features& lastFeatures)
		{
			for (unsigned int i = 0; i < features.size(); ++i)
			{
				unsigned int x = features[i].p2D.x;
				unsigned int y = features[i].p2D.y;

				if (x < roi.x || x > roi.x + roi.width ||
					y < roi.y || y > roi.y + roi.height)
				{
					features.remove(i);
					lastFeatures.remove(i);
					--i;
				}
			}
		}

		static void complement(
			const cv::Mat& gray,
			const cv::Rect& roi,
			const cv::Mat& mask,
			Target& target)
		{
			int numPointsMissing = target.maxNumFeatures - target.features.size();
			if (numPointsMissing > 0) {

				//nullptrifyLastPoints(mask, target.features.points(), minFeatDistance);
				cv::Mat contourMask;
				std::vector<std::vector<cv::Point>> contours;
				contours.push_back(target.contour);
				contourMask = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
				cv::drawContours(contourMask, contours, 0, 255, CV_FILLED);

				std::vector<cv::Point2f> complementaryPoints;

				GFTTExtractor::instance().extract(gray, contourMask, roi, complementaryPoints, numPointsMissing);

				if (complementaryPoints.size() > 0)
					target.features.addPoints(complementaryPoints);
			}
		}

		static void removal(const cv::Mat& mask, Target& target)
		{
			for (unsigned int i = 0; i < target.features.size(); ++i)
			{
				int featureID = Util::index1D(target.features[i].p2D, mask.size());
				if (featureID < 0 ||
					featureID >= mask.cols * mask.rows ||
					mask.data[featureID] != target.contourID + CONTOUR_VISUALIZATION_OFFSET)
				{
					target.features.remove(i);
					target.lastFeatures().remove(i);
					--i;
				}
			}


			for (unsigned int i = 0; i < target.features.size(); ++i)
			{ //then try to find out if there are features too near from each other and remove them
				for (unsigned int j = i + 1; j < target.features.size(); ++j)
				{ //'j = i + 1' because all the previous combinations were already checked
					if (Util::distance4C(target.features[i].p2D, target.features[j].p2D) < MIN_DISTANCE)
					{
						target.features.remove(i);
						target.lastFeatures().remove(i);
						--i;
						break;
					}
				}
			}
		}

		static void updateRelevancesBasedOnVelocity(Features& features, Features& lastFeatures)
		{
			for (unsigned int i = 0; i < features.size(); ++i)
			{ //relevance is a function of the feature velocity
				const auto dist = Util::distanceEuclidean(features[i].p2D, lastFeatures[i].p2D);
				features.setRelevance(dist, i);
			}
		}

		static void updateRelevancesBasedOnAge(Features& features, Features& lastFeatures)
		{
			for (unsigned int i = 0; i < features.size(); ++i)
			{ //relevance is equal to the number of frames that the feature has been tracked
				features.setRelevance(lastFeatures[i].relevance + 1, i);
			}
		}

		static void opticalFlow(const cv::Mat& currGray,
			const cv::Mat& lastGray,
			const cv::Mat& mask,
			Target& target)
		{
			std::vector<uchar> status;
			std::vector<float> err;
			std::vector<cv::Point2f> points;
			std::vector<cv::Point2f> lastPoints;
			lastPoints = target.lastFeatures().points2D();
			/**/
			cv::Size winSize(31, 31);
			cv::TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

			if (lastPoints.size() > 0)
				cv::calcOpticalFlowPyrLK(lastGray, currGray, lastPoints, points, status, err, winSize, 3, termcrit, 0, 0.001);

			target.features.setPoints(points);

			for (unsigned int i = 0; i < target.features.size(); ++i)
			{ //bad features, e.g. non-skin features
				if (!status[i])
				{ //remove the feature as well as its last position
					target.features.remove(i);
					target.lastFeatures().remove(i);
					status.erase(status.begin() + i);
					--i;
				}
			}
		}

		static void follow(const cv::Mat& currGray,
			const cv::Mat& lastGray,
			const cv::Mat& mask,
			Target& target)
		{
			auto succeeded = false;
			auto relevance = NONE;

			if (target.getType() == HAND) relevance = VELOCITY;
			else if (target.getType() == FACE) relevance = AGE;

			//pyramid-based optical flow
			opticalFlow(currGray, lastGray, mask, target);

			//feature removal based on its skin-color and minimum distance
			removal(mask, target);

			if (target.features.size() > 0)
			{
				//set each feature relevance
				if (relevance == VELOCITY) updateRelevancesBasedOnVelocity(target.features, target.lastFeatures());
				else if (relevance == AGE) updateRelevancesBasedOnAge(target.features, target.lastFeatures());

				//calculating the median point
				//calculateMedianPointPan(mask, target, true);
				calculateMedianPoint(target);

				//defining region of interest
				//Util::commonBox(target.medianPoint, mask.size(), 0.2, target.roi);
				Util::boundingBox(target.features.points2D(), mask.size(), ROI_INCREMENT, target.roi);

				//removing roi outliers
				//removeOutliers(target.roi, target.features, target.lastFeatures());

				//relocating the farthest feature of features
				unsigned int numRelocations = target.features.size() * 0.05;
				for (auto i = 0; i < numRelocations; ++i)
					relocate(target.medianPoint, target.features);

				//feature complement
				complement(currGray, target.roi, mask, target);

				succeeded = true;
			}

			target.succeeded = succeeded;
		}

		static void reinitialization(const cv::Mat& gray, cv::Mat& mask, Target& target)
		{
			auto succeeded = false;
			cv::Rect nextROI;

			if (target.lastROI().width > 0 && target.lastROI().height > 0)
			{
				nextROI.width = target.roi.width;
				nextROI.height = target.roi.height;
				nextROI.x = target.roi.x + (target.roi.x - target.lastROI().x);
				nextROI.y = target.roi.y + (target.roi.y - target.lastROI().y);
				Util::limitBox(gray.size(), nextROI);

				complement(gray, target.lastROI(), mask, target);
				complement(gray, target.roi, mask, target);
				complement(gray, nextROI, mask, target);
			}

			if (target.features.size() > 0)
			{
				PanFollower::calculateMedianPoint(target);
				Util::boundingBox(target.features.points2D(), gray.size(), 0, target.roi);
				Util::limitBox(gray.size(), target.roi);

				succeeded = true;
			}

			target.succeeded = succeeded;
		}
	};
}