/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "Util.h"
#include "Corner.h"

namespace haft
{
	class CornersExtractor
	{
	public:
		static void extract(const std::vector<cv::Point>& contour,
			const std::vector<unsigned int>& distances,
			float minCosine,
			int   minCornersGroupSize,
			int   maxCornersGroupSize,
			int   maxDistanceBetweenGroups,
			int   minGreatGroupSize,
			std::vector<Corner>& corners)
		{

			std::vector<Corner> allPointsCorners;
			std::vector<CornersGroup> cornersGroupMap;
			std::vector<Group> groupsGroupMap;

			CornersExtractor::extractAllPointsCorners(contour, distances, minCosine, allPointsCorners);
			CornersExtractor::groupNearbyCorners(allPointsCorners, minCornersGroupSize, maxCornersGroupSize, cornersGroupMap);
			CornersExtractor::groupNearbyCornersGroups(allPointsCorners, cornersGroupMap, maxDistanceBetweenGroups, minGreatGroupSize, groupsGroupMap);
			CornersExtractor::updateCornersGroupData(allPointsCorners, cornersGroupMap, groupsGroupMap);

			Corner corner;
			corner.cos = 0.0;
			int id1, id2;
			for (int i = 0; i < groupsGroupMap.size(); ++i)
			{
				id1 = i;
				for (int j = groupsGroupMap[id1].start; j < groupsGroupMap[id1].start + groupsGroupMap[id1].size(); ++j)
				{
					id2 = j % cornersGroupMap.size();
					corner.point = cornersGroupMap[id2].point;
					corner.dir = cornersGroupMap[id2].direction;
					corners.push_back(corner);
				}
			}
		}

		static void extractAllPointsCorners(const std::vector<cv::Point>& contour,
			const std::vector<unsigned int>& distances,
			float minCosine,
			std::vector<Corner>& corners)
		{

			Corner corner;
			std::vector<int> backPositions;
			std::vector<int> frontPositions;
			const unsigned int numTests = distances.size();
			const unsigned int perimeter = contour.size();
			corners.clear();

			for (int i = 0; i < perimeter; ++i)
			{
				//adding all neighbors positions
				backPositions.clear();
				frontPositions.clear();
				for (int j = 0; j < numTests; ++j)
				{
					backPositions.push_back(i - distances[j]);
					while (backPositions[j] < 0)
						backPositions[j] += perimeter;

					frontPositions.push_back(i + distances[j]);
					while (frontPositions[j] >= perimeter)
						frontPositions[j] -= perimeter;
				}

				bestCorner(contour, i, backPositions, frontPositions, corner);

				if (corner.cos > minCosine)
				{
					corners.push_back(corner);
				}
			}
		}

		static void groupNearbyCorners(const std::vector<Corner>& corners,
			int minGroupSize,
			int maxGroupSize,
			std::vector<CornersGroup>& groups)
		{
			Corner last(0, 0, cv::Point(100000000, 100000000)); //just a far away point to facilitate the creation of the first group
			Corner curr;
			for (int i = 0; i < corners.size(); ++i)
			{
				curr = corners[i];
				//if it is not adjacent to the last corner, then initiate another group
				const unsigned int distance = Util::distance4C(last.point, curr.point);
				if (distance > 2)
				{
					groups.push_back(CornersGroup(i, i, corners.size()));
				}
				else
				{
					groups[groups.size() - 1].end = i;
				}
				last = curr;
			}

			//append the first and last group if conected
			if (groups.size() >= 2)
			{
				if (Util::distance4C(corners[groups[0].start].point, corners[groups[groups.size() - 1].end].point) <= 2)
				{
					groups[0].start = groups[groups.size() - 1].start;
					groups.pop_back();
				}
			}

			//removing small groups, as well is huge ones
			for (int i = 0; i < groups.size(); ++i)
			{
				if (groups[i].size() < minGroupSize || groups[i].size() > maxGroupSize)
				{
					groups.erase(groups.begin() + i);
					--i;
				}
			}
		}

		static void groupNearbyCornersGroups(const std::vector<Corner>& corners,
			const std::vector<CornersGroup>& groupsOfCorners,
			int maxDistanceBetweenGroups,
			int minGroupSize,
			std::vector<Group>& groupsOfGroups)
		{
			bool firstGroup = true;
			CornersGroup lastGroup(0, 0, 0);
			CornersGroup currGroup(0, 0, 0);
			cv::Point currPoint;
			cv::Point lastPoint;
			for (int i = 0; i < groupsOfCorners.size(); ++i)
			{
				currGroup = groupsOfCorners[i];
				currPoint = corners[currGroup.start].point;
				lastPoint = corners[lastGroup.end].point;
				//if it is not close enough from the last corner, then initiate another group
				if (firstGroup || Util::distance4C(lastPoint, currPoint) > maxDistanceBetweenGroups)
				{
					groupsOfGroups.push_back(Group(i, i, groupsOfCorners.size()));
					firstGroup = false;
				}
				else
				{
					groupsOfGroups[groupsOfGroups.size() - 1].end = i;
				}
				lastGroup = currGroup;
			}

			if (groupsOfGroups.size() > 0)
			{
				//append the first and last group if conected
				currPoint = corners[groupsOfCorners[groupsOfGroups[0].start].start].point;
				lastPoint = corners[groupsOfCorners[groupsOfGroups[groupsOfGroups.size() - 1].end].end].point;
				if (groupsOfGroups.size() > 1 && Util::distance4C(lastPoint, currPoint) <= maxDistanceBetweenGroups)
				{
					groupsOfGroups[0].start = groupsOfGroups[groupsOfGroups.size() - 1].start;
					groupsOfGroups.pop_back();
				}
			}

			//removing fake hands with less than 4 fingers and 3 junctions
			for (int i = 0; i < groupsOfGroups.size(); ++i)
			{
				if (groupsOfGroups[i].size() < minGroupSize)
				{
					groupsOfGroups.erase(groupsOfGroups.begin() + i);
					--i;
				}
			}
		}

		static void updateCornersGroupData(const std::vector<Corner>& corners,
			std::vector<CornersGroup>& groupsOfCorners,
			std::vector<Group>& groupsOfGroups)
		{
			//updating direction percentages and mid points of each corner group
			int id1, id2, id3;
			double x, y;
			for (int i = 0; i < groupsOfGroups.size(); ++i)
			{
				id1 = i;
				for (int j = groupsOfGroups[id1].start; j < groupsOfGroups[id1].start + groupsOfGroups[id1].size(); ++j)
				{
					id2 = j % groupsOfCorners.size();

					x = 0;
					y = 0;
					for (int k = groupsOfCorners[id2].start; k < groupsOfCorners[id2].start + groupsOfCorners[id2].size(); ++k)
					{
						id3 = k % corners.size();
						x += corners[id3].point.x;
						y += corners[id3].point.y;

						if (corners[id3].dir > 0)
							groupsOfCorners[id2].direction++;
						else
							groupsOfCorners[id2].direction--;
					}
					x /= groupsOfCorners[id2].size();
					y /= groupsOfCorners[id2].size();
					groupsOfCorners[id2].point = cv::Point(x, y);
					groupsOfCorners[id2].direction /= groupsOfCorners[id2].size();
				}
			}
		}


	private:
		static void bestCorner(const std::vector<cv::Point>& contour,
			int midPointPosition,
			const std::vector<int> backPositions,
			const std::vector<int> frontPositions,
			Corner& corner)
		{
			corner.cos = INT_MIN;
			float cosine;

			const unsigned int numTests = backPositions.size();

			for (int i = 0; i < numTests; ++i)
			{
				cosine = Util::cos(contour[backPositions[i]], contour[midPointPosition], contour[frontPositions[i]]);

				//update best corner
				if (cosine > corner.cos)
				{
					corner.cos = cosine;
					corner.dir = Util::dir(contour[backPositions[i]], contour[midPointPosition], contour[frontPositions[i]]);
					corner.point = contour[midPointPosition];
				}
			}
		}
	};
}