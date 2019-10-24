/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "UI.h"

namespace haft
{	
	//Pinheiro-Rocha-Figueiredo Labeler
	class PRFLabeler
	{
	public:
		static PRFLabeler& instance()
		{
			static PRFLabeler _instance;

			return _instance;
		}

		void label(const cv::Mat& segmentedImage)
		{
			//cleaning class atributes    
			for (auto i = 0; i < labelsVector.size(); ++i) labelsVector[i].clear();
			for (auto i = 0; i < labelsMap.size(); ++i) labelsMap.clear();
			labelsVector.clear();
			labelsMap.clear();
			if (labels != nullptr) delete[]labels;
			if (rtable != nullptr) delete[]rtable;

			const int cols = segmentedImage.cols;
			const int rows = segmentedImage.rows;
			const int size = cols * rows;
			labels = new int[size]; //..................... declaring provisional/representative arrays (NxM/4-sized)
			rtable = new int[size / 4];

			const unsigned int max = cols + 1;
			int *s_queue = new int[max]; //...................... declaring recording and use run data (N/2 +1-sized)
			int *e_queue = new int[max];

			memset(labels, 0, size*sizeof(int));
			memset(rtable, 0, size*sizeof(int) >> 2);
			memset(s_queue, 0, max*sizeof(int));
			memset(e_queue, 0, max*sizeof(int));

			int u = 0; //........................................ useful variables for the first scan
			int v = 0;
			int count = 0;
			int q_in = 0;
			int q_out = 0;

			uchar* data = segmentedImage.data;
			unsigned int index = 0;
			for (unsigned int i = 0; i < rows; ++i)
			{
				for (unsigned int j = 0; j < cols; ++j)
				{
					if (data[index] > 0)
					{
						if (j == 0 || data[index - 1] == 0)
						{
							q_in = (q_in + 1) % max;
							s_queue[q_in] = index;
							labels[index] = ++count;
							rtable[count] = count;
						}
						else
						{
							labels[index] = count;
						}
						e_queue[q_in] = index;
					}
					if (data[index] == 0 && index > 0 && data[index - 1] > 0)
					{
						while (e_queue[q_out] < (s_queue[q_in] - cols - 1))
						{
							q_out = (q_out + 1) % max;
						}
						while (s_queue[q_out] <= (e_queue[q_in] - cols + 1))
						{
							u = labels[index - 1];
							v = labels[s_queue[q_out]];

							merge(rtable, u, v);

							if (e_queue[q_out] <= (e_queue[q_in] - cols))
							{
								q_out = (q_out + 1) % max;
							}
							else
							{
								break;
							}
						}
					}
					index++;
				}
			}


			index = 0;
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					if (labels[index] > 0)
						labelsMap[findSet(rtable, labels[index])] = 1;

					index++;
				}
			}

			//Timer::start();
			index = 0;
			std::map<int, int>::iterator it;

			for (it = labelsMap.begin(); it != labelsMap.end(); ++it)
			{
				it->second = index++;
			}

			//writing labels on image
			index = 0;
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					if (labels[index] > 0)
					{
						const int labelValue = labelsMap[findSet(rtable, labels[index])];

						while (labelsVector.size() <= labelValue)
							labelsVector.push_back(std::vector<unsigned int>());

						labelsVector[labelValue].push_back(index);
					}

					index++;
				}
			}

			delete[]s_queue;
			delete[]e_queue;
		}

		inline std::vector<unsigned int> findLabelPixels(cv::Point2f& point, const cv::Size& imageSize)
		{
			return findLabelPixels(Util::index1D(point, imageSize));
		}

		inline std::vector<unsigned int> findLabelPixels(int index)
		{
			std::vector<unsigned int> result;

			if (labels[index] > 0)
			{
				const int labelSetID = findSet(rtable, labels[index]);
				const int labelMapID = labelsMap[labelSetID];
				result = labelsVector[labelMapID];
			}

			return result;
		}

		inline void removeSmallGroups(cv::Mat& segmentedImage, int minSize)
		{
			std::vector<int> label;
			for (auto i = 0; i < labelsVector.size(); ++i)
			{
				if (labelsVector[i].size() <= minSize) removeSpecificGroup(segmentedImage, i);
			}
		}

		inline void removeSpecificGroup(cv::Mat& segmentedImage, int groupID)
		{
			auto label = labelsVector[groupID];
			const int labelSize = label.size();
			for (auto i = 0; i < labelSize; ++i)
			{
				segmentedImage.data[label[i]] = 0;
			}
		}

	private:
		inline int findSet(int *rtable, int l1)
		{
			if (rtable[l1] != l1)
			{
				rtable[l1] = findSet(rtable, rtable[l1]);
			}
			return rtable[l1];
		}

		inline void merge(int *rtable, int l1, int l2)
		{
			l1 = findSet(rtable, l1);
			l2 = findSet(rtable, l2);
			rtable[l1] = l2;
		}

		std::vector<std::vector<unsigned int>> labelsVector;
		std::map<int, int> labelsMap;
		int *labels;
		int *rtable;

		PRFLabeler() : labels(nullptr), rtable(nullptr) {}

		PRFLabeler(PRFLabeler const&);            //hiding the copy constructor
		PRFLabeler& operator=(PRFLabeler const&); //hiding the assignment operator
	};
}