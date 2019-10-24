/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "UI.h"
#include "Detector.h"
#include "Segmenter.h"
#include "Follower.h"
#include "Labeler.h"
#include "Evaluator.h"

namespace haft
{
	class Tracker
	{
	public:
		static Tracker& getInstance()
		{
			static Tracker instance; // Guaranteed to be destroyed. Instantiated on first use.
			return instance;
		}

		void normalize(std::vector<Target>& targets, int width, int height)
		{
			for (auto&& target : targets)
			{
				target.position = cv::Point2f(
					target.midPoint.x /= width,
					target.midPoint.y /= height);

				target.position.x -= 0.5f;
				target.position.y -= 0.5f;

				target.scale = 
					float(std::min(target.roi.width, target.roi.height)) / 
					std::min(width, height);
			}
		}

		void track(const cv::Mat& frame)
		{
			inputWidth = frame.cols;
			inputHeight = frame.rows;

			cv::cvtColor(frame, currGray, CV_BGR2GRAY);
			debugImage = frame;

			Segmenter::segment(mask, frame);
			Labeler::label(targets, contours, mask, currGray);
			Detector::detect(targets, contours, mask, currGray);
			Follower::follow(targets, currGray, mask, lastGray);
			Evaluator::evaluate(targets, contours, mask, currGray);

			normalize(targets, frame.cols, frame.rows);

			cv::swap(currGray, lastGray);
			
			if (showInput) UI::showImage(frame, "input");
			if (showSegmentation) UI::showImage(mask, "segmentation");
			if (showOutput) UI::showImage(debugImage, "output");
		}

		std::vector<Target> targets;
		bool showInput = false;
		bool showSegmentation = false;
		bool showOutput = true;

	private:
		std::vector<std::vector<cv::Point>> contours;

		cv::Mat mask;
		cv::Mat currGray;
		cv::Mat lastGray;

		int inputWidth;
		int inputHeight;

		Tracker()
		{
			inputWidth = 1;
			inputHeight = 1;
		}; // Constructor? (the {} brackets) are needed here.
		// Dont forget to declare these two. You want to make sure they
		// are unaccessable otherwise you may accidently get copies of
		// your singleton appearing.
		Tracker(Tracker const&);        // Don't Implement
		void operator=(Tracker const&); // Don't implement
	};
};
