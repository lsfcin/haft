/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */


#pragma once

#include <fstream>

#include "Util.h"
#include "PixelClassifier.h"

namespace haft
{
class Histogram
{
public:
	Histogram(unsigned int dimensions = 2, const int& bins1 = 128, const int& range1 = 256,
		const int& bins2 = 128, const int& range2 = 256,
		const int& bins3 = 128, const int& range3 = 256) :
		dimensions(dimensions),
		sum(0), bins1(bins1),
		bins2(bins2), bins3(bins3),
		range1(range1), range2(range2),
		range3(range3)
	{
	}
	/** /
	Histogram(bool (*isColorValid) (float c1, float c2, float c3), bool useC1 = true, bool useC2 = true, bool useC3 = true)
	{
	populate(isColorValid, useC1, useC2, useC3);
	}
	Histogram(const PixelClassifier& classifier, bool useC1 = true, bool useC2 = true, bool useC3 = true)
	{
	populate(classifier, useC1, useC2, useC3);
	}
	//*/

	void save(std::string url)
	{
		std::cout << "saving histogram at: " << url << std::endl;

		std::ofstream file(url.c_str());

		updateSum();

		file << dimensions << " " << sum << " "
			<< bins1 << " " << range1 << " "
			<< bins2 << " " << range2 << " "
			<< bins3 << " " << range3 << std::endl;

		if (dimensions == 1)
			for (unsigned int i = 0; i < bins1; ++i)
				file << data.at<float>(i) << " ";

		if (dimensions == 2)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					file << data.at<float>(i, j) << " ";

		if (dimensions == 3)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					for (unsigned int k = 0; k < bins3; ++k)
						file << data.at<float>(i, j, k) << " ";

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

		file >> dimensions >> sum
			>> bins1 >> range1
			>> bins2 >> range2
			>> bins3 >> range3;

		cv::Mat blankImage = cv::Mat::zeros(1, 1, CV_8UC3);
		populate(blankImage, blankImage);

		if (dimensions == 1)
			for (unsigned int i = 0; i < bins1; ++i)
				file >> data.at<float>(i);

		if (dimensions == 2)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					file >> data.at<float>(i, j);

		if (dimensions == 3)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					for (unsigned int k = 0; k < bins3; ++k)
						file >> data.at<float>(i, j, k);

		file.close();
		std::cout << "loaded." << std::endl;
	}

	inline const int getBins() const
	{
		return std::max({ bins1, bins2, bins3 });
	}

	void populate(const cv::Mat& image,
		const cv::Mat& mask,
		bool useC1 = true,
		bool useC2 = false,
		bool useC3 = false,
		bool accumulate = true)
	{
		div1 = Util::perfectPowerOf2Multiplier(range1, bins1);
		div2 = Util::perfectPowerOf2Multiplier(range2, bins2);
		div3 = Util::perfectPowerOf2Multiplier(range3, bins3);

		dimensions = useC1 + useC2 + useC3;

		if (div1 < 0 || div2 < 0 || div3 < 0)
		{
			std::cerr << "ERROR: The number of bins of used histograms must be a division of the range by power of two." << std::endl;
			exit(0);
		}

		if (dimensions == 1) populateHistogram1D(image, mask, useC1, useC2, useC3, accumulate);
		else if (dimensions == 2) populateHistogram2D(image, mask, useC1, useC2, useC3, accumulate);
		else if (dimensions == 3) populateHistogram3D(image, mask, accumulate);
		else
		{
			std::cerr << "ERROR: Undefined number of histograms dimensions!" << std::endl;
			exit(0);
		}
	}
	void populate(bool(*isColorValid) (float c1, float c2, float c3), bool useC1 = true, bool useC2 = true, bool useC3 = true)
	{
		div1 = Util::perfectPowerOf2Multiplier(range1, bins1);
		div2 = Util::perfectPowerOf2Multiplier(range2, bins2);
		div3 = Util::perfectPowerOf2Multiplier(range3, bins3);

		if (div1 < 0 || div2 < 0 || div3 < 0)
		{
			std::cerr << "ERROR: The number of bins of used histograms must be a division of the range by power of two." << std::endl;
			exit(0);
		}

		dimensions = useC1 + useC2 + useC3;

		if (dimensions == 1) populateHistogram1D(isColorValid, useC1, useC2, useC3);
		else if (dimensions == 2) populateHistogram2D(isColorValid, useC1, useC2, useC3);
		else if (dimensions == 3) populateHistogram3D(isColorValid);
		else
		{
			std::cerr << "ERROR: Undefined number of histograms dimensions!" << std::endl;
			exit(0);
		}
	}
	void populate(const PixelClassifier& classifier, bool useC1 = true, bool useC2 = true, bool useC3 = true)
	{
		div1 = Util::perfectPowerOf2Multiplier(range1, bins1);
		div2 = Util::perfectPowerOf2Multiplier(range2, bins2);
		div3 = Util::perfectPowerOf2Multiplier(range3, bins3);

		if (div1 < 0 || div2 < 0 || div3 < 0)
		{
			std::cerr << "ERROR: The number of bins of used histograms must be a division of the range by power of two." << std::endl;
			exit(0);
		}

		dimensions = useC1 + useC2 + useC3;

		if (dimensions == 1) populateHistogram1D(classifier, useC1, useC2, useC3);
		else if (dimensions == 2) populateHistogram2D(classifier, useC1, useC2, useC3);
		else if (dimensions == 3) populateHistogram3D(classifier);
		else
		{
			std::cerr << "ERROR: Undefined number of histograms dimensions!" << std::endl;
			exit(0);
		}
	}

	bool isPopulated()
	{
		return !data.empty();
	}

	void updateSum()
	{
		sum = 0;
		if (dimensions == 1)
			for (unsigned int i = 0; i < bins1; ++i)
				sum += data.at<float>(i);

		if (dimensions == 2)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					sum += data.at<float>(i, j);

		if (dimensions == 3)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					for (unsigned int k = 0; k < bins3; ++k)
						sum += data.at<float>(i, j, k);
	}

	void makeSumOne()
	{
		updateSum();

		if (sum == 0) sum = 1;

		if (dimensions == 1)
			for (unsigned int i = 0; i < bins1; ++i)
				data.at<float>(i) /= sum;

		if (dimensions == 2)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					data.at<float>(i, j) /= sum;

		if (dimensions == 3)
			for (unsigned int i = 0; i < bins1; ++i)
				for (unsigned int j = 0; j < bins2; ++j)
					for (unsigned int k = 0; k < bins3; ++k)
						data.at<float>(i, j, k) /= sum;

		updateSum();
	}

	void mergeHistogram(const Histogram& inputHist, const float inputWeight)
	{
		if (inputWeight < 1.0f)
		{
			if (dimensions == 1)
				for (unsigned int i = 0; i < bins1; ++i)
					data.at<float>(i) = inputWeight*inputHist.data.at<float>(i) + (1 - inputWeight)*data.at<float>(i);

			if (dimensions == 2)
				for (unsigned int i = 0; i < bins1; ++i)
					for (unsigned int j = 0; j < bins2; ++j)
						data.at<float>(i, j) = inputWeight*inputHist.data.at<float>(i, j) + (1 - inputWeight)*data.at<float>(i, j);

			if (dimensions == 3)
				for (unsigned int i = 0; i < bins1; ++i)
					for (unsigned int j = 0; j < bins2; ++j)
						for (unsigned int k = 0; k < bins3; ++k)
							data.at<float>(i, j, k) = inputWeight*inputHist.data.at<float>(i, j, k) + (1 - inputWeight)*data.at<float>(i, j, k);
		}
		else
			inputHist.data.copyTo(data);

		updateSum();
	}

	void removeLowBins(const float threshold)
	{
		if (dimensions == 1)
		{
			for (int i = 0; i < bins1; ++i)
				if (data.at<float>(i) < float(threshold))
					data.at<float>(i) = 0.f;
		}
		else if (dimensions == 2)
		{
			for (int i = 0; i < bins1; ++i)
				for (int j = 0; j < bins2; ++j)
					if (data.at<float>(i, j) < float(threshold))
						data.at<float>(i, j) = 0.f;
		}
		else if (dimensions == 3)
		{
			for (int i = 0; i < bins1; ++i)
				for (int j = 0; j < bins2; ++j)
					for (int k = 0; k < bins3; ++k)
						if (data.at<float>(i, j, k) < float(threshold))
							data.at<float>(i, j, k) = 0.f;
		}

		makeSumOne();
	}

	cv::Mat calcMean()
	{
		cv::Mat result = cv::Mat::zeros(dimensions, 1, CV_64FC1);

		if (dimensions == 1)
			for (int i = 0; i < bins1; ++i)
				result.at<double>(0) += (i << div1) * data.at<float>(i);

		else if (dimensions == 2)
			for (int i = 0; i < bins1; ++i)
				for (int j = 0; j < bins2; ++j)
				{
					result.at<double>(0) += (i << div1) * data.at<float>(i, j);
					result.at<double>(1) += (j << div2) * data.at<float>(i, j);
				}

		else if (dimensions == 3)
			for (int i = 0; i < bins1; ++i)
				for (int j = 0; j < bins2; ++j)
					for (int k = 0; k < bins2; ++k)
					{
						result.at<double>(0) += (i << div1) * data.at<float>(i, j, k);
						result.at<double>(1) += (j << div2) * data.at<float>(i, j, k);
						result.at<double>(2) += (k << div3) * data.at<float>(i, j, k);
					}

		updateSum();
		result /= sum;
		return result;
	}
	cv::Mat calcCovarianceMatrix(const cv::Mat& mean)
	{
		cv::Mat result = cv::Mat::zeros(dimensions, dimensions, CV_64FC1);
		cv::Mat color = cv::Mat::zeros(dimensions, 1, CV_64FC1);

		if (dimensions == 1)
			for (int i = 0; i < bins1; ++i)
			{
				color.at<double>(0) = (i << div1);
				result += data.at<float>(i) * ((color - mean) * (color - mean).t());
			}

		else if (dimensions == 2)
			for (int i = 0; i < bins1; ++i)
				for (int j = 0; j < bins2; ++j)
				{
					color.at<double>(0) = (i << div1);
					color.at<double>(1) = (j << div2);
					result += data.at<float>(i, j) * ((color - mean) * (color - mean).t());
				}

		else if (dimensions == 3)
			for (int i = 0; i < bins1; ++i)
				for (int j = 0; j < bins2; ++j)
					for (int k = 0; k < bins2; ++k)
					{
						color.at<double>(0) = (i << div1);
						color.at<double>(1) = (j << div2);
						color.at<double>(1) = (k << div3);
						result += data.at<float>(i, j, k) * ((color - mean) * (color - mean).t());
					}

		updateSum();
		result /= sum;
		return result;
	}

	const cv::MatND getHistogram() const
	{
		return data;
	}

	void replaceHistogramData(const cv::MatND& newData)
	{
		if (newData.rows == data.rows &&
			newData.cols == data.cols &&
			newData.type() == data.type())
			data = newData.clone();
	}

	cv::Mat calcCovarianceMatrix()
	{
		cv::Mat mean = calcMean();
		return calcCovarianceMatrix(mean);
	}

	inline unsigned int getDimensions() const
	{
		return dimensions;
	}

	inline float at(int c1) const
	{
		return data.at<float>(c1 >> div1);
	}
	inline float at(int c1, int c2) const
	{
		return data.at<float>(c1 >> div1, c2 >> div2);
	}
	inline float at(int c1, int c2, int c3) const
	{
		return data.at<float>(c1 >> div1, c2 >> div2, c2 >> div2);
	}

	double getSum()
	{
		return sum;
	}

private:
	cv::MatND data;
	unsigned int dimensions;
	double sum; //sum of all bin values
	int bins1, bins2, bins3;
	int range1, range2, range3;
	int div1, div2, div3;

	void populateHistogram1D(const cv::Mat& image,
		const cv::Mat& mask,
		bool useC1 = true,
		bool useC2 = false,
		bool useC3 = false,
		bool accumulate = true)
	{
		float channelRange[] = { 0, range1 };
		const float* ranges = channelRange;
		int channels[] = { 0 };

		if (useC1 && !useC2 && !useC3) channels[0] = 0;
		else if (!useC1 &&  useC2 && !useC3) channels[1] = 0;
		else if (!useC1 && !useC2 &&  useC3) channels[2] = 0;
		else
		{
			std::cerr << "ERROR: wrong number of used coordinates for the histogram!" << std::endl;
			exit(0);
		}

		cv::calcHist(&image, 1, channels, mask, data, 1, &bins1, &ranges, true, accumulate);
	}
	void populateHistogram2D(const cv::Mat& image,
		const cv::Mat& mask,
		bool useC1 = false,
		bool useC2 = true,
		bool useC3 = true,
		bool accumulate = true)
	{
		int histSize[] = { bins1, bins2 };
		float ranges1[] = { 0, range1 };
		float ranges2[] = { 0, range2 };
		const float* ranges[] = { ranges1, ranges2 };
		int channels[] = { 1, 2 };

		if (useC1 &&  useC2 && !useC3) { channels[0] = 0; channels[1] = 1; }
		else if (useC1 && !useC2 &&  useC3) { channels[0] = 0; channels[1] = 2; }
		else if (!useC1 &&  useC2 &&  useC3) { channels[0] = 1; channels[1] = 2; }
		else
		{
			std::cerr << "ERROR: wrong number of used coordinates for the histogram!" << std::endl;
			exit(0);
		}

		cv::calcHist(&image, 1, channels, mask, data, 2, histSize, ranges, true, accumulate);
	}
	void populateHistogram3D(const cv::Mat& image,
		const cv::Mat& mask,
		bool accumulate = true)
	{
		int histSize[] = { bins1, bins2, bins3 };
		float ranges1[] = { 0, range1 };
		float ranges2[] = { 0, range2 };
		float ranges3[] = { 0, range3 };
		const float* ranges[] = { ranges1, ranges2, ranges3 };
		int channels[] = { 0, 1, 2 };
		cv::calcHist(&image, 1, channels, mask, data, 3, histSize, ranges, true, accumulate);
	}

	void populateHistogram1D(bool(*isColorValid) (float c1, float c2, float c3), bool useC1 = true, bool useC2 = false, bool useC3 = false)
	{
		for (unsigned int i = 0; i < bins1; ++i)
		{
			if (useC1 && isColorValid(i << div1, 0, 0)) data.at<float>(i) = 1;
			if (useC2 && isColorValid(0, i << div1, 0)) data.at<float>(i) = 1;
			if (useC3 && isColorValid(0, 0, i << div1)) data.at<float>(i) = 1;
		}
	}
	void populateHistogram2D(bool(*isColorValid) (float c1, float c2, float c3), bool useC1 = true, bool useC2 = true, bool useC3 = false)
	{
		for (unsigned int i = 0; i < bins1; ++i)
		{
			for (unsigned int j = 0; j < bins2; ++j)
			{
				if (useC1 && useC2 && isColorValid(i << div1, j << div2, 0)) data.at<float>(i, j) = 1;
				if (useC1 && useC3 && isColorValid(i << div1, 0, j << div2)) data.at<float>(i, j) = 1;
				if (useC2 && useC3 && isColorValid(0, i << div1, j << div2)) data.at<float>(i, j) = 1;
			}
		}
	}
	void populateHistogram3D(bool(*isColorValid) (float c1, float c2, float c3))
	{
		for (unsigned int i = 0; i < bins1; ++i)
			for (unsigned int j = 0; j < bins2; ++j)
				for (unsigned int k = 0; k < bins3; ++k)
					if (isColorValid(i*div1, j*div2, k*div3)) data.at<float>(i, j, k) = 1;
	}

	void populateHistogram1D(const PixelClassifier& classifier, bool useC1 = true, bool useC2 = false, bool useC3 = false)
	{
		for (unsigned int i = 0; i < bins1; ++i)
		{
			if (useC1 && classifier.classify(i << div1, 0, 0)) data.at<float>(i) = 1;
			if (useC2 && classifier.classify(0, i << div1, 0)) data.at<float>(i) = 1;
			if (useC3 && classifier.classify(0, 0, i << div1)) data.at<float>(i) = 1;
		}
	}
	void populateHistogram2D(const PixelClassifier& classifier, bool useC1 = true, bool useC2 = true, bool useC3 = false)
	{
		for (unsigned int i = 0; i < bins1; ++i)
		{
			for (unsigned int j = 0; j < bins2; ++j)
			{
				if (useC1 && useC2 && classifier.classify(i << div1, j << div2, 0)) data.at<float>(i, j) = 1;
				if (useC1 && useC3 && classifier.classify(i << div1, 0, j << div2)) data.at<float>(i, j) = 1;
				if (useC2 && useC3 && classifier.classify(0, i << div1, j << div2)) data.at<float>(i, j) = 1;
			}
		}
	}
	void populateHistogram3D(const PixelClassifier& classifier)
	{
		for (unsigned int i = 0; i < bins1; ++i)
			for (unsigned int j = 0; j < bins2; ++j)
				for (unsigned int k = 0; k < bins3; ++k)
					if (classifier.classify(i*div1, j*div2, k*div3)) data.at<float>(i, j, k) = 1;
	}
};
}

