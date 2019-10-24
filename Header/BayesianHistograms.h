/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>

namespace haft
{
	
	#define MAX_HUE 180
	enum HISTOGRAMS { BOTH, POSITIVE, NEGATIVE };

	class BayesianHistograms
	{
	public:
		BayesianHistograms(unsigned int numDimensions = 1, float minProbability1 = 0.2f, float minProbability2 = 0.5f,
			const int& bins1 = 64, const int& range1 = 256,
			const int& bins2 = 64, const int& range2 = 256,
			const int& bins3 = 64, const int& range3 = 256) :
			minProbability1(minProbability1), minProbability2(minProbability2), numDimensions(numDimensions),
			positiveCount(0), negativeCount(0),
			bins1(bins1), bins2(bins2),
			bins3(bins3), range1(range1),
			range2(range2), range3(range3), lut(nullptr)
		{
			binIDFactor[0] = 0;
			binIDFactor[1] = 0;
			binIDFactor[2] = 0;
		}

		BayesianHistograms(std::string url, float minProbability1 = 0.2f, float minProbability2 = 0.5f) :
			minProbability1(minProbability1), minProbability2(minProbability2), lut(nullptr)
		{
			binIDFactor[0] = 0;
			binIDFactor[1] = 0;
			binIDFactor[2] = 0;
			load(url);
			pHistSkin = float(positiveCount) / (positiveCount + negativeCount);
			updateTerms1();
			updateTerms2();
		}

		void save(std::string url)
		{
			std::cout << "saving histogram at: " << url << std::endl;

			std::ofstream file(url.c_str());
			file << numDimensions << " " << positiveCount << " " << negativeCount << " "
				<< bins1 << " " << range1 << " "
				<< bins2 << " " << range2 << " "
				<< bins3 << " " << range3 << std::endl;

			if (numDimensions == 1)
				for (unsigned int i = 0; i < bins1; ++i)
				{
					file << positiveHistogram.at<float>(i) << " ";
					file << negativeHistogram.at<float>(i) << " ";
				}

			if (numDimensions == 2)
				for (unsigned int i = 0; i < bins1; ++i)
					for (unsigned int j = 0; j < bins2; ++j)
					{
						file << positiveHistogram.at<float>(i, j) << " ";
						file << negativeHistogram.at<float>(i, j) << " ";
					}

			if (numDimensions == 3)
				for (unsigned int i = 0; i < bins1; ++i)
					for (unsigned int j = 0; j < bins2; ++j)
						for (unsigned int k = 0; k < bins3; ++k)
						{
							file << positiveHistogram.at<float>(i, j, k) << " ";
							file << negativeHistogram.at<float>(i, j, k) << " ";
						}

			file.close();
			std::cout << "saved." << std::endl;
		}

		void load(std::string url, bool accumulate = true)
		{
			std::cout << "loading histogram at: " << url << std::endl;

			std::ifstream file(url.c_str());
			if (!file.good())
			{
				std::cerr << "ERROR: histogram file could not be loaded!" << std::endl;
				system("pause");
				exit(0);
			}
			file >> numDimensions >> positiveCount >> negativeCount
				>> bins1 >> range1
				>> bins2 >> range2
				>> bins3 >> range3;



			cv::Mat blankImage = cv::Mat::zeros(1, 1, CV_8UC3);
			populateHistograms(blankImage, blankImage);

			if (numDimensions == 1)
				for (unsigned int i = 0; i < bins1; ++i)
				{
					file >> positiveHistogram.at<float>(i);
					file >> negativeHistogram.at<float>(i);
				}

			if (numDimensions == 2)
				for (unsigned int i = 0; i < bins1; ++i)
					for (unsigned int j = 0; j < bins2; ++j)
					{
						file >> positiveHistogram.at<float>(i, j);
						file >> negativeHistogram.at<float>(i, j);
					}

			if (numDimensions == 3)
				for (unsigned int i = 0; i < bins1; ++i)
					for (unsigned int j = 0; j < bins2; ++j)
						for (unsigned int k = 0; k < bins3; ++k)
						{
							file >> positiveHistogram.at<float>(i, j, k);
							file >> negativeHistogram.at<float>(i, j, k);
						}


			updateLUT();
			file.close();
			std::cout << "loaded." << std::endl;
		}

		void createFormatedMasks(const cv::Mat& unformatedMask, cv::Mat& positiveFormatedMask, cv::Mat& negativeFormatedMask)
		{
			positiveFormatedMask = cv::Mat::zeros(unformatedMask.rows, unformatedMask.cols, CV_8UC1);
			negativeFormatedMask = cv::Mat::ones(unformatedMask.rows, unformatedMask.cols, CV_8UC1);
			const unsigned int channels = unformatedMask.channels();
			const unsigned int size = unformatedMask.cols * unformatedMask.rows;
			for (unsigned int i = 0; i < size; ++i)
			{
				if (unformatedMask.data[i*channels] != 0)
				{
					positiveFormatedMask.data[i] = 255;
					negativeFormatedMask.data[i] = 0;
					++positiveCount;
				}
				else
				{
					++negativeCount;
				}
			}
			pHistSkin = float(positiveCount) / (positiveCount + negativeCount);
			updateTerms1(); //updating the positive and negative terms used in the isSkinHue method
			updateTerms2();
		}

		void updateLUT()
		{
			if (lut) delete[] lut;
			if (numDimensions == 1)
			{
				lut = new unsigned int[bins1];
				for (int i = 0; i < bins1; ++i)
					lut[i] = 100.f * probability1D(i << div1);

			}
			if (numDimensions == 2)
			{
				lut = new unsigned int[bins1 * bins2];
				int index = 0;
				for (unsigned int j = 0; j < bins2; ++j)
					for (unsigned int i = 0; i < bins1; ++i)
					{
						lut[index] = 100.f * probability2D(i << div1, j << div2);
						++index;
					}
			}
			if (numDimensions == 3)
			{
				lut = new unsigned int[bins1 * bins2 * bins3];
				auto index = 0;
				for (unsigned int k = 0; k < bins3; ++k)
					for (unsigned int j = 0; j < bins2; ++j)
						for (unsigned int i = 0; i < bins1; ++i)
						{
							lut[index] = 100.f * probability3D(i << div1, j << div2, k << div3);
							++index;
						}
			}
		}

		void populateHistograms(const cv::Mat& image, const cv::Mat& mask, HISTOGRAMS activatedHists = BOTH)
		{
			cv::Mat positiveMask, negativeMask;
			createFormatedMasks(mask, positiveMask, negativeMask);

			div1 = Util::perfectPowerOf2Multiplier(range1, bins1);
			div2 = Util::perfectPowerOf2Multiplier(range2, bins2);
			div3 = Util::perfectPowerOf2Multiplier(range3, bins3);

			if (div1 < 0 || div2 < 0 || div3 < 0)
			{
				std::cerr << "ERROR: The number of bins of used histograms must be a division of the range by power of two." << std::endl;
				exit(0);
			}

			if (numDimensions == 1)
			{
				populateHistograms1D(image, positiveMask, negativeMask, activatedHists);
			}
			else if (numDimensions == 2)
			{
				populateHistograms2D(image, positiveMask, negativeMask, activatedHists);
			}
			else if (numDimensions == 3)
			{
				populateHistograms3D(image, positiveMask, negativeMask, activatedHists);
			}
			else
			{
				std::cerr << "ERROR: Undefined number of histograms dimensions!" << std::endl;
				exit(0);
			}
			updateLUT();
		}

		float getMinProbability1() const
		{
			return this->minProbability1;
		}

		float getMinProbability2() const
		{
			return this->minProbability2;
		}

		void setMinProbability1(float minProbability1)
		{
			this->minProbability1 = minProbability1;
			this->updateTerms1();
		}

		void setMinProbability2(float minProbability2)
		{
			this->minProbability2 = minProbability2;
			this->updateTerms2();
		}

		//this probability is calculated based on the following paper,
		//"A Survey on Pixel-Based Skin Color Detection Techniques" of "Vladimir Vezhnevets" 
		inline bool isValid1D(const float& c1, bool lowThreshold = false) const
		{
			const unsigned int binID = c1 * binIDFactor[0];
			if (lowThreshold) return positiveTerm1 * positiveHistogram.at<float>(binID) > negativeTerm1 * negativeHistogram.at<float>(binID);
			else             return positiveTerm2 * positiveHistogram.at<float>(binID) > negativeTerm2 * negativeHistogram.at<float>(binID);
		}
		inline bool isValid2D(const float& c1, const float& c2, bool lowThreshold = false) const
		{
			const unsigned int bin1ID = c1 * binIDFactor[0];
			const unsigned int bin2ID = c2 * binIDFactor[1];
			if (lowThreshold)  return positiveTerm1 * positiveHistogram.at<float>(bin1ID, bin2ID) > negativeTerm1 * negativeHistogram.at<float>(bin1ID, bin2ID);
			else              return positiveTerm2 * positiveHistogram.at<float>(bin1ID, bin2ID) > negativeTerm2 * negativeHistogram.at<float>(bin1ID, bin2ID);
		}
		inline bool isValid3D(const float& c1, const float& c2, const float& c3, bool lowThreshold = false) const
		{
			const unsigned int bin1ID = c1 * binIDFactor[0];
			const unsigned int bin2ID = c2 * binIDFactor[1];
			const unsigned int bin3ID = c3 * binIDFactor[2];
			if (lowThreshold)  return positiveTerm1 * positiveHistogram.at<float>(bin1ID, bin2ID, bin3ID) > negativeTerm1 * negativeHistogram.at<float>(bin1ID, bin2ID, bin3ID);
			else              return positiveTerm2 * positiveHistogram.at<float>(bin1ID, bin2ID, bin3ID) > negativeTerm2 * negativeHistogram.at<float>(bin1ID, bin2ID, bin3ID);
		}

		//validation using lut, considerably faster, please use this instead of the isValidND functions above
		inline bool valid(const unsigned int& c1, const unsigned int& probability) const
		{
			return lut[c1 >> div1] >= probability;
		}
		inline bool valid(const unsigned& c1, const unsigned& c2, const unsigned int& probability) const
		{
			return lut[Util::index1D(c1 >> div1, c2 >> div2, bins1)] >= probability;
		}
		inline bool valid(const unsigned& c1, const unsigned& c2, const unsigned& c3, const unsigned int& probability) const
		{
			return lut[Util::index1D(c1 >> div1, c2 >> div2, c3 >> div3, bins1, bins2)] >= probability;
		}

		inline float probability1D(const float& c1)
		{
			const unsigned int binID = c1 * binIDFactor[0];
			const float pHueSkin = positiveHistogram.at<float>(binID) / positiveCount;
			const float pNotHueSkin = negativeHistogram.at<float>(binID) / negativeCount;
			const float term1 = pHueSkin * pHistSkin;
			return term1 / (term1 + (pNotHueSkin * (1 - pHistSkin)) + 0.000001f);
		}
		inline float probability2D(const float& c1, const float& c2)
		{
			const unsigned int bin1ID = c1 * binIDFactor[0];
			const unsigned int bin2ID = c2 * binIDFactor[1];
			const float pHueSkin = positiveHistogram.at<float>(bin1ID, bin2ID) / positiveCount;
			const float pNotHueSkin = negativeHistogram.at<float>(bin1ID, bin2ID) / negativeCount;
			const float term1 = pHueSkin * pHistSkin;
			return term1 / (term1 + (pNotHueSkin * (1 - pHistSkin)) + 0.000001f);
		}
		inline float probability3D(const float& c1, const float& c2, const float& c3)
		{
			const unsigned int bin1ID = c1 * binIDFactor[0];
			const unsigned int bin2ID = c2 * binIDFactor[1];
			const unsigned int bin3ID = c3 * binIDFactor[2];
			const float pHueSkin = positiveHistogram.at<float>(bin1ID, bin2ID, bin3ID) / positiveCount;
			const float pNotHueSkin = negativeHistogram.at<float>(bin1ID, bin2ID, bin3ID) / negativeCount;
			const float term1 = pHueSkin * pHistSkin;
			return term1 / (term1 + (pNotHueSkin * (1 - pHistSkin)) + 0.000001f);
		}

		inline unsigned int getNumDimensions() const
		{
			return numDimensions;
		}

		cv::MatND positiveHistogram;
		cv::MatND negativeHistogram;

	private:
		float minProbability1;
		float minProbability2;
		unsigned int numDimensions;
		unsigned int positiveCount;
		unsigned int negativeCount;
		int bins1, bins2, bins3;
		int range1, range2, range3;
		int div1, div2, div3;
		unsigned int *lut;          //look up table

		//internal processing
		unsigned int positiveTerm1; //used to optimize the classification of the pixels
		unsigned int negativeTerm1; //idem  
		unsigned int positiveTerm2; //used to optimize the classification of the pixels
		unsigned int negativeTerm2; //idem
		float pHistSkin;
		float binIDFactor[3];

		inline void updateTerms1()
		{
			positiveTerm1 = negativeCount * pHistSkin * (1 - minProbability1);
			negativeTerm1 = positiveCount * (1 - pHistSkin) * minProbability1;
		}

		inline void updateTerms2()
		{
			positiveTerm2 = negativeCount * pHistSkin * (1 - minProbability2);
			negativeTerm2 = positiveCount * (1 - pHistSkin) * minProbability2;
		}

		void populateHistograms1D(const cv::Mat& image, const cv::Mat& positiveMask, const cv::Mat& negativeMask,
			bool accumulate = true, HISTOGRAMS activatedHists = BOTH)
		{
			binIDFactor[0] = float(bins1) / range1;

			float channelRange[] = { 0, range1 };
			const float* ranges = channelRange;
			int channels[] = { 0 };
			if (activatedHists == BOTH || activatedHists == POSITIVE)
				cv::calcHist(&image, 1, channels, positiveMask, positiveHistogram, 1, &bins1, &ranges, true, accumulate);
			if (activatedHists == BOTH || activatedHists == NEGATIVE)
				cv::calcHist(&image, 1, channels, negativeMask, negativeHistogram, 1, &bins1, &ranges, true, accumulate);
		}

		void populateHistograms2D(const cv::Mat& image, const cv::Mat& positiveMask, const cv::Mat& negativeMask,
			bool accumulate = true, HISTOGRAMS activatedHists = BOTH)
		{
			binIDFactor[0] = float(bins1) / range1;
			binIDFactor[1] = float(bins2) / range2;

			int histSize[] = { bins1, bins2 };
			float ranges1[] = { 0, range1 };
			float ranges2[] = { 0, range2 };
			const float* ranges[] = { ranges1, ranges2 };
			int channels[] = { 1, 2 };
			if (activatedHists == BOTH || activatedHists == POSITIVE)
				cv::calcHist(&image, 1, channels, positiveMask, positiveHistogram, 2, histSize, ranges, true, accumulate);
			if (activatedHists == BOTH || activatedHists == NEGATIVE)
				cv::calcHist(&image, 1, channels, negativeMask, negativeHistogram, 2, histSize, ranges, true, accumulate);
		}

		void populateHistograms3D(const cv::Mat& image, const cv::Mat& positiveMask, const cv::Mat& negativeMask,
			bool accumulate = true, HISTOGRAMS activatedHists = BOTH)
		{
			binIDFactor[0] = float(bins1) / range1;
			binIDFactor[1] = float(bins2) / range2;
			binIDFactor[2] = float(bins3) / range3;

			int histSize[] = { bins1, bins2, bins3 };
			float ranges1[] = { 0, range1 };
			float ranges2[] = { 0, range2 };
			float ranges3[] = { 0, range3 };
			const float* ranges[] = { ranges1, ranges2, ranges3 };
			int channels[] = { 0, 1, 2 };
			if (activatedHists == BOTH || activatedHists == POSITIVE)
				cv::calcHist(&image, 1, channels, positiveMask, positiveHistogram, 3, histSize, ranges, true, accumulate);
			if (activatedHists == BOTH || activatedHists == NEGATIVE)
				cv::calcHist(&image, 1, channels, negativeMask, negativeHistogram, 3, histSize, ranges, true, accumulate);
		}
	};
}
