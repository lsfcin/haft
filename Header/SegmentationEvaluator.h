/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

namespace haft
{
	struct SegmentationEvaluation
	{
		double precision;
		double recall;
		double accuracy;
		double specificity;
		double fmeasure;

		SegmentationEvaluation() : precision(0), recall(0), accuracy(0), specificity(0), fmeasure(0) {}
	};

	//TestSet can't be instantiated, it's just a holder for static functions
	class SegmentationEvaluator
	{
	public:
		//main segment method
		inline static void evaluate(const cv::Mat& segmented,
			const cv::Mat& ground,
			SegmentationEvaluation& evaluation)
		{
			const unsigned int size = segmented.cols * segmented.rows;

			//counting true and false positives and negatives
			unsigned int tp = 0;
			unsigned int tn = 0;
			unsigned int fp = 0;
			unsigned int fn = 0;

			for (unsigned int i = 0; i < size; ++i)
			{
				if (segmented.data[i] && ground.data[i]) ++tp;
				if (!segmented.data[i] && !ground.data[i]) ++tn;
				if (segmented.data[i] && !ground.data[i]) ++fp;
				if (!segmented.data[i] && ground.data[i]) ++fn;
			}

			double p = (double)tp + (double)fn; //all positive from ground data
			double n = (double)tn + (double)fp; //all negative from ground data

			evaluation.precision = (double)(tp) / (tp + fp + 0.0000001);
			evaluation.recall = (double)(tp) / (p + 0.0000001);
			evaluation.accuracy = (double)(tp + tn) / (p + n + 0.0000001);
			evaluation.specificity = (double)(tn) / (n + 0.0000001);
			evaluation.fmeasure = (double)(2 * evaluation.recall * evaluation.precision)
				/ (evaluation.recall + evaluation.precision + 0.0000001);
		}

		static void calcAverageAndDeviation(unsigned int size,
			const SegmentationEvaluation* evaluations,
			SegmentationEvaluation& average,
			SegmentationEvaluation& deviation)
		{
			for (int i = 0; i < size; ++i)
			{
				average.recall += evaluations[i].recall;
				average.accuracy += evaluations[i].accuracy;
				average.fmeasure += evaluations[i].fmeasure;
				average.precision += evaluations[i].precision;
				average.specificity += evaluations[i].specificity;
			}

			average.recall /= size;
			average.accuracy /= size;
			average.fmeasure /= size;
			average.precision /= size;
			average.specificity /= size;

			for (int i = 0; i < size; ++i)
			{
				deviation.recall += pow(evaluations[i].recall - average.recall, 2);
				deviation.accuracy += pow(evaluations[i].accuracy - average.accuracy, 2);
				deviation.fmeasure += pow(evaluations[i].fmeasure - average.fmeasure, 2);
				deviation.precision += pow(evaluations[i].precision - average.precision, 2);
				deviation.specificity += pow(evaluations[i].specificity - average.specificity, 2);
			}

			deviation.recall = sqrt(deviation.recall / size);
			deviation.accuracy = sqrt(deviation.accuracy / size);
			deviation.fmeasure = sqrt(deviation.fmeasure / size);
			deviation.precision = sqrt(deviation.precision / size);
			deviation.specificity = sqrt(deviation.specificity / size);
		}

	private:
		SegmentationEvaluator(){}
		~SegmentationEvaluator() {}
		SegmentationEvaluator(SegmentationEvaluator const&);            //hiding the copy constructor
		SegmentationEvaluator& operator=(SegmentationEvaluator const&); //hiding the assignment operator
	};
}