/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once
#include "Target.h"
#include "PanFollower.h"
#include "Refiner.h"


namespace haft
{
	namespace Evaluator
	{
		inline void evaluate(
			std::vector<Target>& targets,
			const std::vector<std::vector<cv::Point>>& contours,
			cv::Mat& mask, 
			const cv::Mat& currGray)
		{
			for (auto i = 0; i >= 0 && i < targets.size(); ++i)
			{
				//if taget couldn't be followed, try then to reinitialize automatically
				if (!targets[i].succeeded)
				{
					PanFollower::reinitialization(currGray, mask, targets[i]);
				}

				//try find a near target that corresponds to the missing one
				if (!targets[i].succeeded)
				{
					for (auto j = 0; j >= 0 && j < targets.size(); ++j)
					{
					}
				}

				//if still not recovered, then remove
				if (!targets[i].succeeded)
				{
					targets.erase(targets.begin() + i);
					--i;
				}

				//merge common targets
				for (auto j = 0; j >= 0 && j < targets.size(); ++j)
				{
					if (i >= 0 && j >= 0 && i != j)
					{
						const auto sameID = targets[i].contourID == targets[j].contourID; //......both targets are within the same contour
						const auto oneIsUnknown = targets[i].getType() == UNKNOWN || targets[j].getType() == UNKNOWN; //at least one of it is unknown
						const auto oneInsideOther = Util::intersectionPercentage(targets[i].roi, targets[j].roi, true) > 0.7; //and its ROIs do overlap at an estabilished percentage
						const auto bothInsideEachOther = Util::intersectionPercentage(targets[i].roi, targets[j].roi, false) > 0.8;

						if (sameID)
						{
							auto removeID = -1;
							if (oneInsideOther && targets[i].getType() == UNKNOWN) removeID = i;
							else if (oneInsideOther && targets[j].getType() == UNKNOWN) removeID = j;
							else if (bothInsideEachOther && targets[i].getType() == HAND &&
								Util::distanceEuclidean(targets[i].medianPoint, targets[j].medianPoint) < 50) removeID = i;
							else if (bothInsideEachOther && targets[j].getType() == HAND &&
								Util::distanceEuclidean(targets[i].medianPoint, targets[j].medianPoint) < 50) removeID = j;
							else if (bothInsideEachOther && targets[i].getType() == FACE &&
								Util::distanceEuclidean(targets[i].medianPoint, targets[j].medianPoint) < 50) removeID = i;
							else if (bothInsideEachOther && targets[j].getType() == FACE &&
								Util::distanceEuclidean(targets[i].medianPoint, targets[j].medianPoint) < 50) removeID = j;

							if (removeID >= 0)
							{
								targets.erase(targets.begin() + removeID);
								--i;
								--j;
								if (i < 0) i = 0;
							}
						}
					}
				}

				//if succeeded in any case above, refine and update the target
				if (i >= 0 && targets[i].succeeded)
				{
					Refiner::refine(mask, targets[i].medianPoint, targets[i]);

					//RENDERING
					cv::Scalar color;
					if (targets[i].getType() == HAND)    color = GREEN;
					if (targets[i].getType() == FACE)    color = BLUE;
					if (targets[i].getType() == UNKNOWN) color = YELLOW;

					if (targets[i].getType() != UNKNOWN)
					{
						UI::showCircle(targets[i].midPoint, BLACK, 10);
						UI::showCircle(targets[i].midPoint, color, 7);
						cv::drawContours(debugImage, contours, targets[i].contourID, color, 1);
					}
					else
					{
						cv::drawContours(debugImage, contours, targets[i].contourID, color, 1);
					}
				}
			}
		}
	}
};