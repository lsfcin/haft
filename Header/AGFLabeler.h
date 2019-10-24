/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>

#include "Util.h"

namespace haft
{
	// Air Guitar Framework Labeler
	class AGFLabeler
	{
	public:
		static AGFLabeler& instance()
		{
			static AGFLabeler _instance;

			return _instance;
		}

		std::vector<unsigned int> makeLabel(unsigned int starterID, const cv::Mat& mask)
		{
			const unsigned int cols = mask.cols;
			const unsigned int rows = mask.rows;
			const unsigned int size = cols * rows;

			std::vector<unsigned int> label;
			std::queue<unsigned int> queue;

			queue.push(starterID);
			label.push_back(starterID);
			markedMap[starterID] = 1;

			unsigned int i;

			while (!queue.empty()) {
				i = queue.front();
				queue.pop();

				bool valid;
				int neighbour;
				for (unsigned int direction = 0; direction < 8; ++direction)
				{
					valid = false;

					switch (direction)
					{
					case 0: //up
						neighbour = i - cols;
						valid = neighbour > 0;
						break;

					case 1: //down
						neighbour = i + cols;
						valid = neighbour < size;
						break;

					case 2: //left
						neighbour = i - 1;
						valid = neighbour % cols > 0;
						break;

					case 3: //right
						neighbour = i + 1;
						valid = neighbour % cols < cols - 1;
						break;

					case 4: //upper-left
						neighbour = i - cols - 1;
						valid = neighbour > 0 && neighbour % cols > 0;
						break;

					case 5: //upper-right
						neighbour = i - cols + 1;
						valid = neighbour > 0 && neighbour % cols < cols - 1;
						break;

					case 6: //bottom-left
						neighbour = i + cols - 1;
						valid = neighbour < size && neighbour % cols > 0;
						break;

					case 7: //bottom-right
						neighbour = i + cols + 1;
						valid = neighbour < size && neighbour % cols < cols - 1;
						break;

					default:
						break;
					}

					if (valid && !markedMap[neighbour] && mask.data[neighbour])
					{
						queue.push(neighbour);
						label.push_back(neighbour);
						markedMap[neighbour] = 1;
					}
				}
			}

			return label;
		}

		void label(const cv::Mat& mask)
		{
			const unsigned int size = mask.cols * mask.rows;
			clean(size);

			for (unsigned int i = 0; i < size; i++)
			{
				if (!markedMap[i] && mask.data[i])
				{
					labels.push_back(makeLabel(i, mask));
				}
			}
		}

		void clean(const unsigned int& size)
		{
			//cleaning last obtained labels
			for (unsigned int i = 0; i < labels.size(); i++)
			{
				labels[i].clear();
			}
			labels.clear();

			//cleaning marked map
			if (markedMap != nullptr) delete markedMap;
			unsigned int labelCount = -1;
			markedMap = new unsigned int[size];
			memset(markedMap, 0, size * sizeof(unsigned int));
		}

		void removeSmallGroups(cv::Mat& mask, unsigned int minSize)
		{
			for (unsigned int i = 0; i < labels.size(); ++i)
			{
				if (labels[i].size() <= minSize) removeSpecificGroup(mask, i);
			}
		}

		void pointExtremities(cv::Mat& mask, unsigned int radius = 3)
		{
			for (unsigned int i = 0; i < labels.size(); ++i)
			{
				UI::showCircle(Util::index2D(labels[i][0], mask.size()), CV_RGB(255, 255, 255), radius, -1, mask);
				UI::showCircle(Util::index2D(labels[i][labels[i].size() - 1], mask.size()), CV_RGB(255, 255, 255), radius, -1, mask);
			}
		}

		inline void removeSpecificGroup(cv::Mat& mask, unsigned int groupID)
		{
			std::vector<unsigned int> label = labels[groupID];
			const unsigned int labelSize = label.size();
			for (unsigned int i = 0; i < labelSize; ++i)
			{
				mask.data[label[i]] = 0;
			}
		}

	private:
		std::vector<std::vector<unsigned int>> labels;
		unsigned int* markedMap;

		AGFLabeler() : markedMap(nullptr) {}
		AGFLabeler(AGFLabeler const&) = delete;            //hiding the copy constructor
		AGFLabeler& operator=(AGFLabeler const&) = delete; //hiding the assignment operator
	};
}