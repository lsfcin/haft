/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "Features.h"

//UI can't be instantiated, it's just a holder for static functions
namespace haft
{
#define WHITE  CV_RGB(255, 255, 255)
#define BLACK  CV_RGB(0, 0, 0)
#define RED    CV_RGB(255, 0, 0)
#define GREEN  CV_RGB(0, 255, 0)
#define BLUE   CV_RGB(0, 0, 255)
#define YELLOW CV_RGB(255, 255, 0)
#define PINK   CV_RGB(255, 0, 255)
#define CYAN   CV_RGB(0, 255, 255)
#define ORANGE CV_RGB(255, 125, 0)

	static cv::Mat debugImage;
	class UI
	{
	public:
		static void combineImages(
			const cv::Mat& input1,
			const cv::Mat& input2,
			cv::Mat& combined)
		{
			auto k = 0;
			for (auto i = 0; i < combined.rows; ++i)
			{
				for (auto j = 0; j < combined.cols; ++j)
				{
					if (j < input1.cols)
					{
						combined.at<cv::Vec3b>(i, j) = input1.at<cv::Vec3b>(i, j);
						k = 0;
					}
					else
					{
						combined.at<cv::Vec3b>(i, j) = input2.at<cv::Vec3b>(i, k);
						k++;
					}
				}
			}
		}

		inline static void mapVar2Key(char keyUP, char keyDOWN, float& variable,
			const float& increment = 1.f, bool print = false,
			const float& min = 0.f, const float& max = 255.f)
		{
			auto key = cv::waitKey(10);
			if (char(key) == keyUP   && variable < max) variable += increment;
			if (char(key) == keyDOWN && variable > min) variable -= increment;
			if (print) std::cout << "value: " << variable;
		}

		/*inline static void mapVar2Trackbar(float& variable, float increment = 1.f, unsigned int max = 255, std::string barName = "TrackBar", std::string windowName = "Image")
		{
			const auto scale = 1 / increment;
			static int tempVar = variable * scale;
			mapVar2Trackbar(tempVar, max * scale, barName, windowName);
			variable = float(tempVar) / scale;
		}*/

		inline static void mapVar2Trackbar(int& variable, unsigned int max = 255, std::string barName = "TrackBar", std::string windowName = "Image")
		{
			cv::createTrackbar(barName, windowName, &variable, max);
		}

		static void showImage(const cv::Mat& image, std::string windowName = "Image")
		{
			cv::namedWindow(windowName.c_str(), 1);
			cv::imshow(windowName.c_str(), image);
		}

		static void showHistogram1D(const cv::Mat& histogram, std::string windowName = "Histogram1D")
		{
			cv::Mat normalizedHist;
			const unsigned int binW = 2;
			cv::normalize(histogram, normalizedHist, 0, 255, CV_MINMAX);
			cv::Mat image = cv::Mat::zeros(200, normalizedHist.rows * binW, CV_8UC3);
			cv::Mat buf(1, normalizedHist.size, CV_8UC3);
			for (unsigned int i = 0; i < normalizedHist.rows; i++)
				buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / normalizedHist.rows), 255, 255);

			cv::cvtColor(buf, buf, CV_HSV2BGR);

			for (unsigned int i = 0; i < normalizedHist.rows; i++)
			{
				const unsigned int colHeight = cv::saturate_cast<int>(normalizedHist.at<float>(i) * image.rows / 255);
				rectangle(image,
					cv::Point(i * binW, image.rows),
					cv::Point((i + 1) * binW, image.rows - colHeight),
					cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
			}
			showImage(image, windowName);
		}

		static void showHistogram2D(const cv::Mat& histogram, 
			std::string windowName = "Histogram2D",
			const unsigned int scale = 4)
		{
			double maxVal = 0;
			cv::minMaxLoc(histogram, 0, &maxVal, 0, 0);

			const unsigned int bins1 = histogram.cols;
			const unsigned int bins2 = histogram.rows;
			cv::Mat histImg = cv::Mat::zeros(bins1*scale, bins2*scale, CV_8UC3);

			for (unsigned int b2 = 0; b2 < bins2; b2++)
			{
				for (unsigned int b1 = 0; b1 < bins1; b1++)
				{
					float binVal = histogram.at<float>(b2, b1);
					unsigned int intensity = cvRound(binVal * 255 / maxVal);
					cv::rectangle(histImg, cv::Point(b2*scale, b1*scale),
						cv::Point((b2 + 1)*scale - 1, (b1 + 1)*scale - 1),
						cv::Scalar::all(intensity),
						CV_FILLED);
				}
			}
			cv::imshow(windowName, histImg);
		}

		inline static void showPointsIDs(const std::vector<unsigned int>& pointsIDs,
			const cv::Scalar& color = GREEN,
			float alpha = 0.7f, float highlight = 1.1f,
			cv::Mat& image = debugImage)
		{
			const unsigned int size = pointsIDs.size();
			const unsigned int numChannels = image.channels();
			for (unsigned int i = 0; i < size; ++i)
			{
				const unsigned int currID = pointsIDs[i] * numChannels;

				for (unsigned int j = 0; j < numChannels; ++j)
				{
					const int tempColor = ((image.data[currID + j] * alpha) +
						(color[j] * (1.f - alpha))) * highlight;
					if (tempColor > 255)
						image.data[currID + j] = 255;
					else
						image.data[currID + j] = tempColor;
				}
			}
		}

		inline static void showCircle(const cv::Point2f& point,
			const cv::Scalar& color = GREEN,
			unsigned int size = 1, int thickness = -1,
			cv::Mat& image = debugImage)
		{
			cv::circle(image, point, size, color, thickness);
		}

		inline static void showCircles(const cv::Point2f* points, unsigned int nPoints,
			const cv::Scalar& color = GREEN,
			unsigned int size = 1, int thickness = -1,
			cv::Mat& image = debugImage)
		{
			for (unsigned int i = 0; i < nPoints; ++i)
				showCircle(points[i], color, size, thickness, image);
		}

		inline static void showCircles(const std::vector<cv::Point2f>& points,
			const cv::Scalar& color = GREEN,
			unsigned int size = 1, int thickness = -1,
			cv::Mat& image = debugImage)
		{
			for (unsigned int i = 0; i < points.size(); ++i)
				showCircle(points[i], color, size, thickness, image);
		}

		inline static void showCircles(const std::vector<cv::Point>& points,
			const cv::Scalar& color = GREEN,
			unsigned int size = 1, int thickness = -1,
			cv::Mat& image = debugImage)
		{
			for (unsigned int i = 0; i < points.size(); ++i)
				showCircle(points[i], color, size, thickness, image);
		}

		inline static void showCircles(const Features& features,
			const cv::Scalar& color = GREEN,
			unsigned int size = 1, int thickness = -1,
			cv::Mat& image = debugImage)
		{
			float maxRelevance = 0;
			int mostRelevant = -1;
			cv::Scalar tempColor = color;

			for (unsigned int i = 0; i < features.size(); ++i)
			{
				if (features[i].relevance > maxRelevance)
				{
					maxRelevance = features[i].relevance;
					mostRelevant = i;
				}
			}

			for (unsigned int i = 0; i < features.size(); ++i)
			{
				const float term = (float)maxRelevance / 1.5;
				tempColor.val[0] *= features[i].relevance / term;
				tempColor.val[1] *= features[i].relevance / term;
				tempColor.val[2] *= features[i].relevance / term;

				//const int radius = min((double)features[i].relevance, 8.0);
				showCircle(features[i].p2D, color, size, thickness, image);
				/*if(i == mostRelevant)
				  showCircle(features[i].p2D, color, size + 1, thickness);
				  else
				  showCircle(features[i].p2D, color, size, thickness);*/
			}
		}

		inline static void showLine(const cv::Point2f& p1, const cv::Point2f& p2,
			const cv::Scalar& color = GREEN,
			unsigned int thickness = 1,
			cv::Mat& image = debugImage)
		{
			cv::line(image, p1, p2, color, thickness);
		}

		inline static void showRectangle(const cv::Rect& rect,
			const cv::Scalar& color = GREEN,
			unsigned int thickness = 1,
			cv::Mat& image = debugImage)
		{
			cv::rectangle(image, rect, color, thickness);
		}

		inline static void showEllipse(const cv::RotatedRect& rotatedBOX,
			const cv::Scalar& color = GREEN,
			unsigned int thickness = 1,
			cv::Mat& image = debugImage)
		{
			cv::ellipse(image, rotatedBOX, color, thickness);
		}

	private:
		UI() {}
		~UI() {}
		UI(UI const&);            //hiding the copy constructor
		UI& operator=(UI const&); //hiding the assignment operator
	};
}
