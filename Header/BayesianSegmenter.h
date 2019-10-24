/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <fstream>
#include <queue>

#include <opencv2/opencv.hpp>

#include "Globals.h"
#include "UI.h"
#include "BayesianHistograms.h"

//#include "GMMClassifier.h"
//#include "LUTClassifier.h"
#include "HSVSobottkaPitas.h"
#include "YCrCbGarciaTziritas.h"
#include "HSVRGBClassifier.h"
#include "YCrCbHSVClassifier.h"
#include "BayesianPixelClassifier.h"
#include "SingleGaussianClassifier.h"


namespace haft
{

#define STARTER_POTENTIAL 3
#define GOOD_SPREADER_POTENTIAL 2
#define BAD_SPREADER_POTENTIAL 1
#define USE_QUEUE_STD 1
#define USE_STACK_STD 0
#define USE_QUEUE_MINE 0
#define USE_STACK_MINE 0

	//first == index, second == propagation potential
	typedef std::pair<unsigned int, unsigned int> PixelIDPotential;

	class BayesianSegmenter
	{
	public:
		static BayesianSegmenter& instance()
		{
			static BayesianSegmenter _instance;

			return _instance;
		}

		void segment(const cv::Mat& image, cv::Mat& mask, BayesianHistograms& histograms, bool lowThreshold = false)
		{
			cv::Mat converted;
			if (histograms.getNumDimensions() == 1) cv::cvtColor(image, converted, CV_BGR2HSV);   //done for convenience
			else if (histograms.getNumDimensions() == 2) cv::cvtColor(image, converted, CV_BGR2YCrCb); //histograms 1D means HSV, 2D means YCC 
			else if (histograms.getNumDimensions() == 3) converted = image;                            //and 3D means BGR

			const unsigned int size = converted.channels() * converted.cols * converted.rows;

			unsigned int c1, c2, c3;
			unsigned int j = 0;
			for (unsigned int i = 0; i < size; i += 3)
			{

				c1 = converted.data[i];
				c2 = converted.data[i + 1];
				c3 = converted.data[i + 2];

				if (isSkin(c1, c2, c3, histograms, lowThreshold))
				{
					mask.data[j] += 255;
				}
				++j;
			}
		}

		void segmentByNeighbourhood(const cv::Mat& imageBGR, const cv::Mat& edges, cv::Mat& mask)
		{
			const unsigned int channels = 3;
			const unsigned int cols = imageBGR.cols;
			const unsigned int rows = imageBGR.rows;
			const unsigned int greySize = cols * rows;
			const unsigned int bgrSize = channels * greySize;

			cv::Mat imageYCC, imageHSV;
			cv::cvtColor(imageBGR, imageYCC, CV_BGR2YCrCb);
			cv::cvtColor(imageBGR, imageHSV, CV_BGR2HSV);

			unsigned int* visited = new unsigned int[greySize];
			memset(visited, 0, greySize * sizeof(unsigned int));

#if USE_QUEUE_STD
			std::queue<PixelIDPotential> container;
#elif USE_STACK_STD
			std::stack<PixelIDPotential> container;
#elif USE_QUEUE_MINE
			Queue<PixelIDPotential> container(greySize);
#elif USE_STACK_MINE
			Stack<PixelIDPotential> container(greySize);
#endif

			unsigned int r, g, b, y, cr, cb, h, s, v;
			unsigned int j = 0, potential, position;
			int up, down, left, right;
			for (unsigned int i = 0; i < bgrSize; i += channels)
			{
				if (!visited[j])
				{
					b = imageBGR.data[i]; g = imageBGR.data[i + 1]; r = imageBGR.data[i + 2];
					y = imageYCC.data[i]; cr = imageYCC.data[i + 1]; cb = imageYCC.data[i + 2];
					h = imageHSV.data[i]; s = imageHSV.data[i + 1]; v = imageHSV.data[i + 2];
					potential = skinPotential(r, g, b, y, cr, cb, h, s, v);

					if (potential >= STARTER_POTENTIAL) //starter pixel found
					{
						container.push(PixelIDPotential(j, potential));
						visited[j] = 1;

						while (!container.empty())
						{
#if USE_STACK_STD
							auto pixel = container.top();
#elif USE_STACK_MINE
							auto pixel = container.top();
#else
							auto pixel = container.front();
#endif
							container.pop();

							position = pixel.first;
							potential = pixel.second;

							mask.data[position] = 55 + std::min(int(potential) << 1, 200);
							visited[position] = 1;

							if (potential && !edges.data[position]) //if it is an edge pixel, then do not spread
							{ //trying to spread skin pixels
								up = pixel.first - imageBGR.cols;
								down = pixel.first + imageBGR.cols;
								left = pixel.first - 1;
								right = pixel.first + 1;
								if (down < greySize)      validatePixel(down, channels, potential, imageBGR, imageYCC, imageHSV, visited, container);
								if (right % cols > 0)     validatePixel(right, channels, potential, imageBGR, imageYCC, imageHSV, visited, container);
								if (left % cols < cols - 1) validatePixel(left, channels, potential, imageBGR, imageYCC, imageHSV, visited, container);
								if (up > 0)               validatePixel(up, channels, potential, imageBGR, imageYCC, imageHSV, visited, container);
							}
						}
					}
				}
				++j;
			}

			delete[] visited;
		}
		void generateHistogramsFromSetOfImages(BayesianHistograms& histograms,
			unsigned int numDimensions = 1, float minProbability = 0.5f,
			const int& bins1 = 64, const int& range1 = 256,
			const int& bins2 = 64, const int& range2 = 256,
			const int& bins3 = 64, const int& range3 = 256)
		{
			histograms = BayesianHistograms(numDimensions, minProbability, bins1, range1, bins2, range2, bins3, range3);

			//std::ifstream skinList   (haft::Globals::resourcesRootUrl +"/Skin Ground Truth/Voxar Database/skin-list.txt");
			//std::ifstream maskList   (haft::Globals::resourcesRootUrl +"/Skin Ground Truth/Voxar Database/mask-list.txt");
			std::ifstream skinList(haft::Globals::resourcesRootUrl + "/Skin Ground Truth/Jones Database/skin-list.txt");
			std::ifstream nonSkinList(haft::Globals::resourcesRootUrl + "/Skin Ground Truth/Jones Database/non-skin-list.txt");
			std::string imageFileName, skinURL, maskURL;
			cv::Mat bgrImage, convertedImage, mask;

			//SKIN IMAGES
			std::cout << "adding skin images to histograms." << std::endl;
			while (!skinList.eof())
			{
				//skinList >> skinURL;
				//maskList >> maskURL;
				skinList >> imageFileName;

				//skinURL = "Resource/Skin Ground Truth/Voxar Database/" + skinURL + ".png";
				//maskURL = "Resource/Skin Ground Truth/Voxar Database/" + maskURL + ".png";
				skinURL = haft::Globals::resourcesRootUrl + "/Skin Ground Truth/Jones Database/" + imageFileName + ".jpg";
				maskURL = haft::Globals::resourcesRootUrl + "/Skin Ground Truth/Jones Database/" + imageFileName + ".pbm";

				bgrImage = cv::imread(skinURL);
				mask = cv::imread(maskURL);
				if (numDimensions == 1) cv::cvtColor(bgrImage, convertedImage, CV_BGR2HSV);
				if (numDimensions == 2) cv::cvtColor(bgrImage, convertedImage, CV_BGR2YCrCb);
				if (numDimensions == 3) convertedImage = bgrImage;
				histograms.populateHistograms(convertedImage, mask);
			}
			skinList.close();
			//maskList.close();

			//SAVING HISTOGRAMS
			if (numDimensions == 1) histograms.save(haft::Globals::resourcesRootUrl + "Skin Histograms/Recently Saved/hue histograms.hst");
			if (numDimensions == 2) histograms.save(haft::Globals::resourcesRootUrl + "Skin Histograms/Recently Saved/cbcr histograms.hst");
			if (numDimensions == 3) histograms.save(haft::Globals::resourcesRootUrl + "Skin Histograms/Recently Saved/rgb histograms.hst");

			//RENDERING
			if (numDimensions == 1)
			{
				UI::showHistogram1D(histograms.positiveHistogram, "pos");
				UI::showHistogram1D(histograms.negativeHistogram, "neg");
			}
			if (numDimensions == 2)
			{
				UI::showHistogram2D(histograms.positiveHistogram, "pos");
				UI::showHistogram2D(histograms.negativeHistogram, "neg");
			}
		}

		BayesianHistograms histogramsHSV, histogramsYCC, histogramsBGR;

	private:
		cv::Mat image;
		cv::Mat mask;

		BayesianSegmenter()
		{
			histogramsHSV = BayesianHistograms(haft::Globals::resourcesRootUrl + "/Skin Histograms/Used/VOXAR hue 90 histograms.hst", 0.01, 0.43);
			histogramsYCC = BayesianHistograms(haft::Globals::resourcesRootUrl + "/Skin Histograms/Used/VOXAR cbcr 64 histograms.hst", 0.3, 0.5);
			histogramsBGR = BayesianHistograms(haft::Globals::resourcesRootUrl + "/Skin Histograms/Used/VOXAR rgb 16 histograms.hst", 0.05, 0.7);
		}

		inline bool isSkin(const unsigned int& c1, const unsigned int& c2, const unsigned int& c3,
			const BayesianHistograms& histograms, bool lowThreshold)
		{
			if (lowThreshold)
			{
				if (histograms.getNumDimensions() == 1) return histograms.valid(c1, 1) && c2 > 30 && c2 < 170 && c3 > 40 && c3 < 230;
				else if (histograms.getNumDimensions() == 2) return histograms.valid(c2, c3, 30) && c1 > 40 && c1 < 210;
				else if (histograms.getNumDimensions() == 3) return histograms.valid(c1, c2, c3, 11);
				else return false;
			}
			else
			{
				if (histograms.getNumDimensions() == 1) return histograms.valid(c1, 43) && c2 > 35 && c2 < 140 && c3 > 110 && c3 < 190;
				else if (histograms.getNumDimensions() == 2) return histograms.valid(c2, c3, 50) && c1 > 60 && c1 < 190;
				else if (histograms.getNumDimensions() == 3) return histograms.valid(c1, c2, c3, 70);
				else return false;
			}
		}

		//google's patent US 8.055.067 B2 RGB modified to support saturation from HSV model
		inline bool RGBGoogleModified(const unsigned int& b, const unsigned int& g, const unsigned int& r, const unsigned int& s)
		{
			const unsigned int k = (s + 5) >> 2;

			return (r > g + k && r > b + k && g > 0 && /*r/b < 2.75 &&*/ r / g < 1.5);
		}

		inline unsigned int skinPotential(const unsigned int& r, const unsigned int& g, const unsigned int& b,
			const unsigned int& y, const unsigned int& cr, const unsigned int& cb,
			const unsigned int& h, const unsigned int& s, const unsigned int& v)
		{
			unsigned int result = 0;
			/*const bool hsv = (s > 35 && s < 140 && v > 110 && v < 190 && histogramsHSV.valid(h, 43));
			const bool ycc = (y > 60 && y < 190 && histogramsYCC.valid(cr, cb, 50));

			if(hsv) result += spreadPotential;
			if(ycc) result += spreadPotential;*/

			if ((s > 35 && s < 140 && v > 110 && v < 190 && histogramsHSV.valid(h, 43)) ||
				(y > 60 && y < 190 && histogramsYCC.valid(cr, cb, 50)))
				//(y > 40 && y < 210 && histogramsYCC.valid(cr, cb, 30)))
				result = STARTER_POTENTIAL;

			if (result == 0)
			{
				if (y > 40 && y < 210 && histogramsYCC.valid(cr, cb, 30)) result = GOOD_SPREADER_POTENTIAL;
				else if ((s > 30 && s < 170 && v > 40 && v < 230 && histogramsHSV.valid(h, 1)) ||
					(histogramsBGR.valid(b, g, r, 11))) result = BAD_SPREADER_POTENTIAL;
			}

			return result;
		}

		//if(RGBGoogle(b, g, r, false)) result = 1;
		/*if(RGBGoogleModified(b, g, r, s)) result = 1;

		else */

		inline void validatePixel(const unsigned int& index, const unsigned int& channels, const unsigned int& fatherPotential,
			const cv::Mat& imageBGR, const cv::Mat& imageYCC, const cv::Mat& imageHSV,
			unsigned int* visited,
#if USE_QUEUE_STD
			std::queue<PixelIDPotential>& container)
#elif USE_STACK_STD
			std::stack<PixelIDPotential>& container)
#elif USE_QUEUE_MINE
			Queue<PixelIDPotential>& container)
#elif USE_STACK_MINE
			Stack<PixelIDPotential>& container)
#endif
		{
			if (!visited[index])
			{
				//if skin
				float r, g, b, y, cr, cb, h, s, v;
				unsigned int i = index * channels;
				b = imageBGR.data[i]; g = imageBGR.data[i + 1]; r = imageBGR.data[i + 2];
				y = imageYCC.data[i]; cr = imageYCC.data[i + 1]; cb = imageYCC.data[i + 2];
				h = imageHSV.data[i]; s = imageHSV.data[i + 1]; v = imageHSV.data[i + 2];
				unsigned int childPotential = skinPotential(r, g, b, y, cr, cb, h, s, v);

				if (childPotential >= STARTER_POTENTIAL)
				{
					container.push(PixelIDPotential(index, childPotential + fatherPotential));
				}
				else if (childPotential >= GOOD_SPREADER_POTENTIAL)
				{
					const double logfather = log(double(fatherPotential));
					const double factor = std::min(logfather / 4, 1.0);
					container.push(PixelIDPotential(index, factor * fatherPotential * 2));
				}
				else if (childPotential >= BAD_SPREADER_POTENTIAL)
				{
					const double logfather = log(double(fatherPotential));
					const double factor = std::min(logfather / 5, 0.99);
					const double potential = std::min(double(fatherPotential), 20.);
					container.push(PixelIDPotential(index, factor * potential));//max(fatherPotential - 1, 0.)));
				}
				visited[index] = 1;
			}
		}

		BayesianSegmenter(BayesianSegmenter const&);            //hiding the copy constructor
		BayesianSegmenter& operator=(BayesianSegmenter const&); //hiding the assignment operator
	};
}
