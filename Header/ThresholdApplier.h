/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>
#include <math.h>
#include <string.h>

#include <opencv2/opencv.hpp>

#include "Segmenter.h"
#include "ContourExtractor.h"

#include "UI.h"
#include "Util.h"

namespace haft
{
	class ThresholdApplier
	{
	public:
		static void calcPobj(const cv::Mat& histogram, const int& threshold, float& pobj)
		{
			pobj = 0;
			for (int i = 0; i < threshold; ++i)
				pobj += histogram.at<float>(i);
		}
		static void calcPbg(const cv::Mat& histogram, const int& threshold, float& pbg)
		{
			const int size = histogram.rows * histogram.cols;
			pbg = 0;
			for (int i = threshold; i < size; ++i)
				pbg += histogram.at<float>(i);
		}

		static void calcRidlerCalvardMobj(const cv::Mat& histogram, const int& threshold, const float& pobj, float& mobj)
		{
			mobj = 0;
			for (int i = 0; i < threshold; ++i)
				mobj += (i + 1)*histogram.at<float>(i);

			mobj /= (pobj + 0.000001f);
		}
		static void calcRidlerCalvardMbg(const cv::Mat& histogram, const int& threshold, const float& pbg, float& mbg)
		{
			const int size = histogram.rows * histogram.cols;

			mbg = 0;
			for (int i = threshold; i < size; ++i)
				mbg += (i + 1 - threshold)*histogram.at<float>(i);

			mbg /= (pbg + 0.000001f);
		}

		static void calcPortesDeAlbuquerqueHobj(const cv::Mat& histogram, const int& threshold, const float& pobj, float& hobj, double q)
		{
			hobj = 0;
			float sum = 0;

			for (int i = 0; i < threshold; ++i)
				sum += pow(static_cast<double>(histogram.at<float>(i) / (pobj + 0.00001f)), q);

			hobj = (1 - sum) / (q - 1 + 0.000001f);
		}
		static void calcPortesDeAlbuquerqueHbg(const cv::Mat& histogram, const int& threshold, const float& pbg, float& hbg, double q)
		{
			hbg = 0;
			float sum = 0;

			const int size = histogram.rows * histogram.cols;

			for (int i = threshold; i < size; ++i)
				sum += pow(static_cast<double>(histogram.at<float>(i) / (pbg + 0.00001f)), q);

			hbg = (1 - sum) / (q - 1 + 0.000001f);
		}

		static void calcRidlerCalvardThreshold(const cv::Mat& input, const cv::Mat& histogram, int& threshold)
		{
			threshold = 0;

			float minArg = INT_MAX;
			float pobj, pbg, mobj, mbg, arg;

			const int size = histogram.cols * histogram.rows;

			for (int i = 0; i < size; ++i)
			{
				calcPobj(histogram, i, pobj);
				calcPbg(histogram, i, pbg);
				calcRidlerCalvardMobj(histogram, i, pobj, mobj);
				calcRidlerCalvardMbg(histogram, i, pbg, mbg);

				arg = (mobj + mbg) / 2;

				if (arg < minArg && mobj > 0.0001f && mbg > 0.0001f)
				{
					minArg = arg;
					threshold = i;
				}
			}
		}
		static void calcPortesDeAlbuquerqueThreshold(const cv::Mat& input, const cv::Mat& histogram, int& threshold, double q = 1.0)
		{
			threshold = 0;

			float maxArg = -1;
			float pobj, pbg, hobj, hbg, arg;

			const int size = histogram.cols * histogram.rows;

			for (int i = 0; i < size; ++i)
			{
				calcPobj(histogram, i, pobj);
				calcPbg(histogram, i, pbg);
				calcPortesDeAlbuquerqueHobj(histogram, i, pobj, hobj, q);
				calcPortesDeAlbuquerqueHbg(histogram, i, pbg, hbg, q);

				arg = hobj + hbg + ((1 - q) * hobj * hbg);

				if (arg > maxArg)
				{
					maxArg = arg;
					threshold = i;
				}
			}
		}

		static void calcHistSum(const cv::Mat& histogram, double& sum)
		{
			const int size = histogram.rows * histogram.cols;
			sum = 0;
			for (int i = 0; i < size; ++i)
				sum += histogram.at<float>(i);
		}
		static void calcHistMax(const cv::Mat& histogram, double& max)
		{
			const int size = histogram.rows * histogram.cols;
			max = 0;
			for (int i = 0; i < size; ++i)
				if (histogram.at<float>(i) > max)
					max = histogram.at<float>(i);
		}
		static void calcGrayHistogram(const cv::Mat& input, cv::Mat& histogram, bool normalized = true)
		{
			int          L = 256;
			int          histSize[] = { L };
			int          histChannels[] = { 0 };
			float        histRange[] = { 0, L };
			const float* histRanges[] = { histRange };
			cv::calcHist(&input, 1, histChannels, cv::Mat(), histogram, 1, histSize, histRanges);

			if (normalized)
			{
				//normalize hist to make its sum equal to 1 (100%)
				double histSum, histMax;
				calcHistSum(histogram, histSum);
				calcHistMax(histogram, histMax);
				double roof4NormalizedHist = histMax / (histSum + 1); //CARE! this +1 should not exist
				cv::normalize(histogram, histogram, 0, roof4NormalizedHist, CV_MINMAX, CV_32F);

				calcHistSum(histogram, histSum);
				CVGUI::showHistogram1D(histogram);
			}
		}

		static void applyGlobalThreshold(const cv::Mat& input, cv::Mat& output)
		{
			const int size = input.cols * input.rows;

			cv::Mat histogram;
			calcGrayHistogram(input, histogram);

			int threshold;
			//calcRidlerCalvardThreshold      (input, histogram, threshold);
			calcPortesDeAlbuquerqueThreshold(input, histogram, threshold, 0.5);

			for (int i = 0; i < size; ++i)
			{
				if (input.data[i] > threshold)
					output.data[i] = 255;
				else
					output.data[i] = 0;
			}
		}
		static void applyLocalThreshold(const cv::Mat& input, cv::Mat& output)
		{
			cv::Mat integral;
			cv::integral(input, integral, CV_32F);

			int threshold;

			int windowSide = 30;
			int windowArea = windowSide * windowSide;
			int windowHalfSize = windowSide / 2;
			int d = windowHalfSize;
			int w = input.cols;
			int h = input.rows;

			//handy variables
			float sum, mean, deviation, value; //deviation => mean deviation, value = img value
			float ul, ur, dl, dr; //ul = up-left, ur = up-right, dl = down-left, dr = down-right
			float k = 0.06;

			for (int y = 0; y < input.rows; ++y)
			{
				for (int x = 0; x < input.cols; ++x)
				{
					if (Util::isInside(x + d - 1, y - d, w, h)) ur = integral.at<float>(y - d, x + d - 1);
					else                                   ur = 0;
					if (Util::isInside(x - d, y - d, w, h)) ul = integral.at<float>(y - d, x - d);
					else                                   ul = 0;
					if (Util::isInside(x + d - 1, y + d - 1, w, h)) dr = integral.at<float>(y + d - 1, x + d - 1);
					else                                   dr = 0;
					if (Util::isInside(x - d, y + d - 1, w, h)) dl = integral.at<float>(y + d - 1, x - d);
					else                                   dl = 0;

					sum = (dr + ul) - (dl + ur);
					mean = sum / windowArea;
					value = input.at<unsigned char>(y, x);
					deviation = value - mean;
					threshold = mean * (1 + k * (deviation / (1 - deviation) - 1));

					if (input.at<unsigned char>(y, x) > threshold) output.at<unsigned char>(y, x) = 255;
					else                                          output.at<unsigned char>(y, x) = 0;
				}
			}
		}

		static void applyThreshold(const cv::Mat& input, cv::Mat& output)
		{
			debugImage = input;

			cv::Mat gray;
			cv::cvtColor(input, gray, CV_BGR2GRAY);
			cv::cvtColor(input, output, CV_BGR2GRAY);

			//applyGlobalThreshold(gray, output);
			applyLocalThreshold(gray, output);
		}
	};
}