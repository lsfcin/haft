/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <math.h>

#include "Histogram.h"
#include "DatabaseReader.h"
#include "PixelClassifier.h"

namespace haft
{	
	class SingleGaussianClassifier : public PixelClassifier
	{
	private:
		int conversion;
		Histogram hist;
		Histogram nskin;
		cv::Mat mean;
		cv::Mat covariance;

		cv::Mat notskinMean;
		cv::Mat notskinCovaciance;

		cv::Mat invCovariance;
		cv::Mat invNotSkinCovariance;

		bool useC1;
		bool useC2;
		bool useC3;

	public:
		SingleGaussianClassifier(DatabaseReader& database,
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
			for (int i = 0; i < size; ++i)
			{
				// reading code
				cv::Mat input, ground;
				database.grabNext(input, ground);
				cv::Mat notground = cv::Mat::ones(ground.rows, ground.cols, ground.type()) - ground;

				cv::Mat converted = input.clone();
				if (conversion >= 0)
					cv::cvtColor(input, converted, conversion);

				hist.populate(converted, ground, useC1, useC2, useC3);
				nskin.populate(converted, notground, useC1, useC2, useC3);
			}

			notskinMean = nskin.calcMean();
			notskinCovaciance = nskin.calcCovarianceMatrix();
			invNotSkinCovariance = notskinCovaciance.inv();

			mean = hist.calcMean();
			covariance = hist.calcCovarianceMatrix(mean);
			invCovariance = covariance.inv();
		}

		inline float classify(uchar c1, uchar c2, uchar c3) const
		{
			//put color in a cv::Mat according to classifier dimension and coordinates
			double d = useC1 + useC2 + useC3;
			cv::Mat color = cv::Mat::zeros(mean.rows, 1, CV_64FC1);
			if (d == 1)
			{
				if (useC1) color.at<double>(0) = c1;
				if (useC2) color.at<double>(0) = c2;
				if (useC3) color.at<double>(0) = c3;
			}
			else if (d == 2)
			{
				if (useC1 && useC2) { color.at<double>(0) = c1; color.at<double>(1) = c2; }
				if (useC1 && useC3) { color.at<double>(0) = c1; color.at<double>(1) = c3; }
				if (useC2 && useC3) { color.at<double>(0) = c2; color.at<double>(1) = c3; }
			}
			else if (d == 3)
			{
				color.at<double>(0) = c1;
				color.at<double>(1) = c2;
				color.at<double>(2) = c3;
			}

			//main calc
			float probability;

			cv::Mat meanMahalanobisExpr = (color - notskinMean).t() * invNotSkinCovariance * (color - notskinMean);
			cv::Mat mahalanobisExpr = (color - mean).t() * invCovariance * (color - mean);
			double meanMahalanobis = meanMahalanobisExpr.at<double>(0);
			double mahalanobis = mahalanobisExpr.at<double>(0);
		
			probability = 1 - (mahalanobis / (meanMahalanobis * 2));
		
			return probability;
		}

		inline int getBGRConversion() const override
		{ return conversion; }
	};
}