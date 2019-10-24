/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "Histogram.h"
#include "DatabaseReader.h"
#include "PixelClassifier.h"

namespace haft
{
	class BayesianPixelClassifier : public PixelClassifier
	{
private:
	int conversion;
	Histogram pHist;
	Histogram nHist;

	bool useC1;
	bool useC2;
	bool useC3;

public:
	BayesianPixelClassifier(
		DatabaseReader& database,
		float dbUsedForTraining = 0.05f,
		int conversion = CV_BGR, //CV_BGR means no conversion is needed
		bool useC1 = true,
		bool useC2 = true,
		bool useC3 = true) : conversion(conversion), useC1(useC1), useC2(useC2), useC3(useC3)
	{
		initialize(database, dbUsedForTraining, useC1, useC2, useC3);
	}
	void initialize(DatabaseReader& database,
		float dbUsedForTraining = 0.05f,
		bool useC1 = true,
		bool useC2 = true,
		bool useC3 = true)
	{
		const unsigned int size = database.getSize() * dbUsedForTraining;
		for (auto i = 0; i < size; ++i)
		{
			// reading code
			cv::Mat input, ground;
			database.grabNext(input, ground);
			cv::Mat segmented = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
			cv::Mat notground = cv::Mat::ones(ground.rows, ground.cols, ground.type()) - ground;

			auto converted = input.clone();
			if (conversion >= 0)
				cv::cvtColor(input, converted, conversion);

			pHist.populate(converted, ground, useC1, useC2, useC3);
			nHist.populate(converted, notground, useC1, useC2, useC3);
		}

		if (size != 0)
		{
			//this is performed to overcome precision problems
			pHist.makeSumOne();
			nHist.makeSumOne();
		}
	}

	inline void update(
		const cv::Mat& input,
		const cv::Mat& positiveMask,
		const cv::Mat& negativeMask,
		const float learningRate,
		const float binsThreshold = 0.01f) override
	{	
		const unsigned int dimension = useC1 + useC2 + useC3;

		if (!pHist.getHistogram().empty() && !nHist.getHistogram().empty())
		{
			Histogram pHistInput;
			Histogram nHistInput;

			// create histograms to input images
			pHistInput.populate(input, positiveMask, useC1, useC2, useC3);
			nHistInput.populate(input, negativeMask, useC1, useC2, useC3);

			// smoothing on input histogram
			const unsigned int kSize = 5;

			cv::Mat output;
			cv::GaussianBlur(pHistInput.getHistogram(), output, cv::Size(kSize, kSize), kSize, kSize);
			pHistInput.replaceHistogramData(output);
			cv::GaussianBlur(nHistInput.getHistogram(), output, cv::Size(kSize, kSize), kSize, kSize);
			nHistInput.replaceHistogramData(output);

			pHistInput.makeSumOne();
			nHistInput.makeSumOne();

			// merge the new on old histogram
			pHist.mergeHistogram(pHistInput, learningRate);
			nHist.mergeHistogram(nHistInput, learningRate);
		}
		else
		{
			// create a histogram with the real time maks
			pHist.populate(input, positiveMask, useC1, useC2, useC3);
			nHist.populate(input, negativeMask, useC1, useC2, useC3);
		}

		// show the new histogram
		cv::Mat posHistImg;
		cv::Mat negHistImg;
		cv::Mat posFinalHistImg;
		cv::Mat negFinalHistImg;

		if (dimension == 1)
		{
			haft::Util::getHistogram1DImg(pHist.getHistogram(), posHistImg);
			haft::Util::getHistogram1DImg(nHist.getHistogram(), negHistImg);
			
			haft::Util::createHistogramImg(posHistImg, posFinalHistImg, conversion);
			haft::Util::createHistogramImg(negHistImg, negFinalHistImg, conversion);
		}	
		else if (dimension == 2)
		{
			haft::Util::getHistogram2DImg(pHist.getHistogram(), posHistImg);
			haft::Util::getHistogram2DImg(nHist.getHistogram(), negHistImg);

			haft::Util::createHistogramImg(posHistImg, posFinalHistImg, conversion);
			haft::Util::createHistogramImg(negHistImg, negFinalHistImg, conversion);
		}

		// remove low bins from histogram
		//removeLowBins(binsThreshold);

		// fit ellipse region of histogram
		ellipseOnHistogram(posFinalHistImg, posHistImg);
		ellipseOnHistogram(negFinalHistImg, negHistImg);

		cv::imshow("Positive Histogram", posFinalHistImg);
		cv::imshow("Negative Histogram", negFinalHistImg);
	}

	inline void removeLowBins(const float threshold = 0.01)
	{
		pHist.removeLowBins(threshold);
		nHist.removeLowBins(threshold);
	}

	inline void ellipseOnHistogram(cv::Mat& histogramImage, cv::Mat& input)
	{
		const unsigned int ratio = 3;
		const unsigned int lowThreshold = 73;

		cv::Mat edge;
		cv::Mat thresholdOutput;
		cv::Mat convertedInput;

		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		convertedInput = cv::Mat::zeros(input.cols, input.rows, CV_8UC1);

		for (auto i = 0; i < input.rows; ++i)
		{
			for (auto j = 0; j < input.cols; ++j)
			{
				if (input.at<cv::Vec3b>(i, j)[0] > 50) convertedInput.at<uchar>(i, j) = 255;

				else convertedInput.at<uchar>(i, j) = 0;
			}
		}

		cv::Canny(convertedInput, edge, lowThreshold, lowThreshold*ratio, 3);
		edge.convertTo(thresholdOutput, CV_8U);

		cv::findContours(thresholdOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		std::vector<cv::RotatedRect> minEllipse(contours.size());

		for (auto i = 0; i < contours.size(); ++i)
			if (contours[i].size() >= 5) // Condition to fitEllipse
				minEllipse[i] = cv::fitEllipse(cv::Mat(contours[i]));

		// Draw ellipses
		cv::Mat drawing = cv::Mat::zeros(thresholdOutput.size(), CV_8UC3);
		
		for (auto i = 0; i < contours.size(); i++) 
			ellipse(histogramImage, minEllipse[i], cv::Scalar(255, 0, 255), 2, 8);
	}

	inline cv::MatND getPosHistogram()
	{
		return pHist.getHistogram();
	}

	inline cv::MatND getNegHistogram()
	{
		return nHist.getHistogram();
	}

	inline float classify(uchar c1, uchar c2, uchar c3) const override
	{
		float pc = 0, nc = 0.000001;

		auto dimensions = useC1 + useC2 + useC3;

		if (dimensions == 1)
		{
			if (useC1) { pc = pHist.at(c1); nc = nHist.at(c1); }
			else if (useC2) { pc = pHist.at(c2); nc = nHist.at(c2); }
			else if (useC3) { pc = pHist.at(c3); nc = nHist.at(c3); }
		}
		else if (dimensions == 2)
		{
			if (useC1 && useC2) { pc = pHist.at(c1, c2); nc = nHist.at(c1, c2); }
			else if (useC1 && useC3) { pc = pHist.at(c1, c3); nc = nHist.at(c1, c3); }
			else if (useC2 && useC3) { pc = pHist.at(c2, c3); nc = nHist.at(c2, c3); }
		}
		else if (dimensions == 3)
		{
			pc = pHist.at(c1, c2, c3); nc = nHist.at(c1, c2, c3);
		}

		auto probability = pc / (pc + nc);

		return probability;
	}

		inline int getBGRConversion() const override
	{ 
		return conversion;
	}
	inline bool readyToSegment()
	{
		return (pHist.isPopulated() && nHist.isPopulated());
	}
};
}