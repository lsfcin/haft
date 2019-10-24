/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace haft
{
	class Feature
	{
	public:
		Feature() {}
		Feature(cv::Point2f p2D, float relevance) : p2D(p2D), relevance(relevance) {}

		cv::Point2f p2D;
		float relevance;
	};

	class Features
	{
	public:
		Features() {}
		Features(std::vector<cv::Point2f>& points) : points(points)
		{
			initializeRelevances(points.size());
		}

		inline void push_back(cv::Point2f point)
		{
			points.push_back(point);
			relevances.push_back(1);
		}

		inline void push_back(Feature feature)
		{
			points.push_back(feature.p2D);
			relevances.push_back(feature.relevance);
		}

		inline unsigned int size() const
		{
			auto size = points.size();
			return size;
		}

		inline std::vector<cv::Point2f>& points2D()
		{
			return points;
		}

		inline void setRelevance(float relevance, unsigned int index)
		{
			relevances[index] = relevance;
		}

		inline void setPoint(cv::Point2f& p, unsigned int index)
		{
			points[index] = p;
		}

		inline void setPoints(std::vector<cv::Point2f>& newPoints)
		{
			points = newPoints;
			initializeRelevances(points.size());
		}

		inline void addPoints(std::vector<cv::Point2f>& newPoints)
		{
			const unsigned int oldSize = points.size();
			const unsigned int newSize = oldSize + newPoints.size();

			points.resize(newSize);

			for (unsigned int i = oldSize; i < newSize; ++i)
			{
				points[i] = newPoints[i - oldSize];
				relevances.push_back(1);
			}
		}

		inline void initializeRelevances(unsigned int size)
		{
			relevances.resize(points.size());
			for (unsigned int i = 0; i < size; ++i)
				relevances[i] = 1;
		}

		inline void setFeature(const Feature newFeature, unsigned int index)
		{
			points[index] = newFeature.p2D;
			relevances[index] = newFeature.relevance;
		}

		inline const Feature operator[] (unsigned int index) const
		{
			Feature result;
			result.p2D = points[index];
			result.relevance = relevances[index];
			return result;
		}

		inline void remove(unsigned int index)
		{
			points.erase(points.begin() + index);
			relevances.erase(relevances.begin() + index);
		}

		inline Features& Features::operator = (const Features& newFeatures)
		{
			if (this == &newFeatures)
				return *this;

			points.clear();
			relevances.clear();
			for (unsigned int i = 0; i < newFeatures.size(); ++i) push_back(newFeatures[i]);
			return *this;
		}

		inline void resize(unsigned int newSize)
		{
			points.resize(newSize);
			relevances.resize(newSize);
		}

		inline void clear()
		{
			points.clear();
			relevances.clear();
		}

		void sort()
		{
			quicksort(0, size() - 1);
		}

		void quicksort(unsigned int first, unsigned int last)
		{
			unsigned int middle;
			if (first < last)
			{
				middle = partition(first, last);
				quicksort(first, middle);  // sort first section
				quicksort(middle + 1, last); // sort second section
			}
			return;
		}

		unsigned int partition(unsigned int first, unsigned int last)
		{
			Feature temp;
			Feature x = (*this)[first];
			unsigned int i = first - 1;
			unsigned int j = last + 1;
			do
			{
				do
				{
					--j;
				} while (x.relevance > (*this)[j].relevance);

				do
				{
					++i;
				} while (x.relevance < (*this)[i].relevance);

				if (i < j)
				{
					temp = (*this)[i];
					setFeature((*this)[j], i);
					setFeature(temp, j);
				}
			} while (i < j);
			return j; // returns middle subscript  
		}

	private:
		std::vector<cv::Point2f> points;
		std::vector<float> relevances;
	};
}
