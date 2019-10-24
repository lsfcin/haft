/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "BayesianSegmenter.h"
#include "UI.h"
#include "PixelClassifier.h"
#include "FixedFunctionClassifier.h"
#include "Util.h"

//Segmenter can't be instantiated, it's just a holder for static functions
namespace haft
{
	class Segmenter
	{
	public:

		// output needs to fit the input resolution
		// and also to store a byte per pixel
		static void checkOutput(const cv::Mat& input, cv::Mat& output)
		{
			if (
				output.cols != input.cols ||
				output.rows != input.rows ||
				output.type() != CV_8UC1)
			{
				output = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
			}
		}

		inline static void segment(
			cv::Mat& mask,
			const cv::Mat& frame,
			cv::Mat& currGray = cv::Mat()) // gray image may be provided in onder to use edges on the expansion region growing classifier
		{
			checkOutput(frame, mask);

			mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			cv::Mat edges = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);

			/**/
			if (currGray.rows == mask.rows && currGray.cols == mask.cols)
			{
				static auto c1 = 40; static auto c2 = 70; static auto minSize = 20;
				cv::Canny(currGray, edges, c1, c2, 3, true);
				cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1); //dilate all contours, if is desired to remove some of them, use the line below instead of this
			}
			//ContourManager::extract(edges, 0, minSize, 2); //remove small contours, and redraw the other ones with line size 2
			/** /
			UI::mapVar2Trackbar(minSize, 1000, "ms", "edges");
			UI::mapVar2Trackbar(c1, 1000, "c1", "edges");
			UI::mapVar2Trackbar(c2, 1000, "c2", "edges");
			UI::showImage(edges, "edges");//*/

			/**/
			static auto option = 0;
			static auto lowThreshold = 1;
			UI::mapVar2Trackbar(option, 4, "seg", "mask");
			UI::mapVar2Trackbar(lowThreshold, 1, "lt", "mask");

			if (option == 0)
			{
				BayesianSegmenter::instance().segmentByNeighbourhood(frame, edges, mask);
			}
			if (option == 1)
			{
				BayesianSegmenter::instance().segment(frame, mask, BayesianSegmenter::instance().histogramsBGR, lowThreshold);
			}
			if (option == 2)
			{
				BayesianSegmenter::instance().segment(frame, mask, BayesianSegmenter::instance().histogramsHSV, lowThreshold);
			}
			if (option == 3)
			{
				BayesianSegmenter::instance().segment(frame, mask, BayesianSegmenter::instance().histogramsYCC, lowThreshold);
			}
			if (option == 4)
			{
				BayesianSegmenter::instance().segment(frame, mask, BayesianSegmenter::instance().histogramsBGR, lowThreshold);
				BayesianSegmenter::instance().segment(frame, mask, BayesianSegmenter::instance().histogramsHSV, lowThreshold);
				BayesianSegmenter::instance().segment(frame, mask, BayesianSegmenter::instance().histogramsYCC, lowThreshold);
			}
			if (option == 5)
			{
				static FixedFunctionClassifier classifier(RGBGoogleFixed, CV_BGR);
				Segmenter::segment(frame, mask, classifier);
			}
			if (option == 6)
			{
				static FixedFunctionClassifier classifier(RGBKovacetalFixed, CV_BGR);
				Segmenter::segment(frame, mask, classifier);
			}
			if (option == 7)
			{
				static FixedFunctionClassifier classifier(RGBGomezMoralesFixed, CV_BGR);
				Segmenter::segment(frame, mask, classifier);
			}
			if (option == 8)
			{
				cv::Mat hsv;
				cv::cvtColor(frame, hsv, CV_BGR2HSV);
				static FixedFunctionClassifier classifier(HSVOpenCVASDFixed, CV_BGR2HSV);
				Segmenter::segment(hsv, mask, classifier);
			}
			if (option == 9)
			{
				cv::Mat hsv;
				cv::cvtColor(frame, hsv, CV_BGR2HSV);
				static FixedFunctionClassifier classifier(HSVTsekeridouPitasFixed, CV_BGR2HSV);
				Segmenter::segment(hsv, mask, classifier);
			}
		}

		inline static void segment(
			const cv::Mat& input,
			cv::Mat& output,
			const PixelClassifier& classifier,
			const float& closure = 0.0f)
		{
			checkOutput(input, output);			

			const unsigned int channels = input.channels();
			const auto size = input.cols * input.rows * channels;

			unsigned int j = 0;
			uchar c1, c2, c3;
			for (unsigned int i = 0; i < size; i += channels)
			{
				c1 = input.data[i];
				c2 = input.data[i + 1];
				c3 = input.data[i + 2];

				auto classification = classifier.classify(c1, c2, c3);

				if (classification >= closure)
					output.data[j] = 255 * classification;

				++j;
			}
		}
		
		// this method just segment a region intern of a bouding box
		inline static void segment(
			const cv::Mat& input,
			cv::Mat& output,
			const PixelClassifier& classifier,
			cv::Rect& boundingBox,
			const float& closure = 0.0f)
		{
			checkOutput(input, output);

			const unsigned int rows = input.rows;
			const unsigned int cols = input.cols;

			unsigned int k = 0;
			uchar c1, c2, c3;
			for (unsigned int i = 0; i < rows; ++i)
			{
				for (unsigned int j = 0; j < cols; ++j)
				{
					if ((j >= boundingBox.x && j <= (boundingBox.x + boundingBox.width)) &&
						(i >= boundingBox.y && i <= (boundingBox.y + boundingBox.height)))
					{
						c1 = input.at<cv::Vec3b>(i, j)[0];
						c2 = input.at<cv::Vec3b>(i, j)[1];
						c3 = input.at<cv::Vec3b>(i, j)[2];

						auto classification = classifier.classify(c1, c2, c3);

						if (classification >= closure)
							output.data[k] = 255 * classification;
					}
						
					++k;
				}
			}
		}

		//expansion based classifier
		//segmentation considers classifier1 as seed and then expands to classifier2 results
		//works only for BGR images
		inline static void segment(
			const cv::Mat& input,
			cv::Mat& output,
			const PixelClassifier& classifier1,
			const PixelClassifier& classifier2,
			const cv::Mat& edges)
		{
			checkOutput(input, output);

			const unsigned int channels = input.channels();
			const unsigned int width = input.cols;
			const unsigned int height = input.rows;
			const auto pixels = width * height;
			const auto size = pixels * channels;

			unsigned int j = 0;
			uchar b, g, r;

			int current;
			int neighbors[4] = { 0, 0, 0, 0 };
			double potential = 0;

			Queue<int> boosters(pixels);
			Queue<int> dampers(pixels);

			cv::Mat map = cv::Mat::ones(input.rows, input.cols, CV_8UC1);

			for (unsigned int i = 0; i < size; i += channels)
			{
				if (map.data[j])
				{
					b = input.data[i];
					g = input.data[i + 1];
					r = input.data[i + 2];

					if (classifier1.classify(b, g, r))
					{
						potential = 1;
						boosters.push(j);

						//mark
						map.data[j] = 0;

						while (potential > 0 && !(boosters.empty() && dampers.empty()))
						{
							//current always prefers boosters among dampers
							if (!boosters.empty())
								current = boosters.pop();
							else
								current = dampers.pop();

							output.data[current] = std::min(100 + int(potential / 80), 255);

							//calc current neighbors
							neighbors[0] = std::max<int>(current - width, 0);      //up
							neighbors[1] = std::min<int>(current + width, pixels); //down
							neighbors[2] = std::max<int>(current - 1, 0);          //left
							neighbors[3] = std::min<int>(current + 1, pixels);     //left

							//visit them, mark, and add if possible add to the 
							for (auto n = 0; n < 4; ++n)
							{
								auto id = neighbors[n] * 3;
								if (map.data[neighbors[n]])
								{
									b = input.data[id];
									g = input.data[id + 1];
									r = input.data[id + 2];

									if (edges.data[current])
									{
										potential -= 1;
									}
									else if (classifier1.classify(b, g, r))
									{
										boosters.push(neighbors[n]);
										potential += 1;

										map.data[neighbors[n]] = 0;
									}
									else if (classifier2.classify(b, g, r))
									{
										dampers.push(neighbors[n]);
										potential -= 1;

										map.data[neighbors[n]] = 0;
									}
									else
									{
										potential -= 1;
									}
								}
							}
						}
					}
				}
				++j;
			}
		}

	private:
		Segmenter() {}
		~Segmenter() {}
		Segmenter(Segmenter const&);            //hiding the copy constructor
		Segmenter& operator=(Segmenter const&); //hiding the assignment operator
	};
}