/*
* This file is subject to the terms and conditions defined in
* file 'LICENSE.txt', which is part of this source code package.
*/

#include "Detector.h"

bool haft::Detector::detectFaces = true;
bool haft::Detector::detectHands = true;

void haft::Detector::detect(
	std::vector<Target>& targets,
	const std::vector<std::vector<cv::Point>>& contours,
	const cv::Mat& mask,
	const cv::Mat& currGray)
{
	const int numTargets = targets.size();
	for (unsigned int i = 0; i < contours.size(); ++i)
	{
		auto alreadyFollowing = false;

		cv::Rect contourROI;
		Util::boundingBox(contours[i], currGray.size(), 20, contourROI);
		Util::limitBox(currGray.size(), contourROI);

		//finding and updating contours with already followed targets
		for (unsigned int j = 0; j < numTargets; ++j)
		{
			//if target j has the same i contour, than the current contour is already being used for a target
			if (targets[j].contourID == i)
			{
				//if target is of UNKNOWN type, try to identify it
				if (targets[j].getType() == UNKNOWN)
				{
					//first try to find out if the target is a hand, since it is very fast it can be performed every frame
					std::vector<Hand> hands;
					if (detectHands) HandDetector::detect(contours[i], hands);
					if (hands.size() > 0)
					{
						for (auto k = 0; k < hands.size(); ++k)
						{
							targets[j].increaseHandProbability(0.51f);
							targets[j].roi = hands[k].rect;
							targets[j].midPoint = hands[k].palm;
						}
					}
					else
					{
						//if it is not a hand try to find if it is a face, only if the probability is 0 or more, it is a mechanism to jump some frames
						if (targets[j].getFaceProbability() >= 0)
						{
							std::vector<cv::Rect> faces;
							if (detectFaces) AdaboostDetector::instance().detect(currGray(contourROI), faces, FACE);
							if (faces.size() > 0)
							{
								targets[j].increaseFaceProbability(0.34f);
								targets[j].roi = faces[0];
								targets[j].roi.x += contourROI.x;
								targets[j].roi.y += contourROI.y;
							}
							else
							{
								//if a face is not detected, set the probability to be negative, in order to jump some frames
								targets[j].decreaseFaceProbability(1.f);
							}
						}
						//but in case the probability is below zero just keep incrementing it
						else
						{
							targets[j].increaseFaceProbability(0.05f);
						}
					}
				}

				alreadyFollowing = true;
				break;
			}
		}

		//creating new targets with the not yet followed contours
		//if the i contour is not being followed yet, start following it
		if (!alreadyFollowing)
		{
			auto detected = false;

			auto target = Target();
			target.setROI(contourROI);

			std::vector<Hand> hands;
			if (detectHands) HandDetector::detect(contours[i], hands);
			for (auto k = 0; k < hands.size(); ++k)
			{
				target.increaseHandProbability(0.51f);
				target.roi = hands[k].rect;
				target.midPoint = hands[k].palm;

				detected = true;
			}

			Util::limitBox(mask.size(), target.roi);
			target.contour = contours[i];
			target.contourID = i;

			cv::Mat contourMask;
			std::vector<std::vector<cv::Point>> tempContours;
			tempContours.push_back(target.contour);
			contourMask = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
			cv::drawContours(contourMask, tempContours, 0, 255, CV_FILLED);
			GFTTExtractor::instance().extract(currGray, contourMask, target);
			target.succeeded = true;
			targets.push_back(target);

			cv::Scalar color;
			if (target.getType() == HAND)    color = GREEN;
			if (target.getType() == FACE)    color = BLUE;
			if (target.getType() == UNKNOWN) color = YELLOW;
		}
	}
}