/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "Segmenter.h"
#include "HSVSobottkaPitas.h"
#include "YCrCbGarciaTziritas.h"
#include "BayesianPixelClassifier.h"
#include "SingleGaussianClassifier.h"
#include "LUTManager.h"

namespace haft
{
	class BGRLumminanceTransformer
	{
	public:
		static BGRLumminanceTransformer& getInstance()
		{
			static BGRLumminanceTransformer instance; // Guaranteed to be destroyed. Instantiated on first use.
			return instance;
		}

		void adapt(const cv::Mat& input, cv::Mat& output)
		{
			static cv::Vec3i center;
			center[0] = 128;
			center[1] = 128;
			center[2] = 128;

			static int rD = 20;
			static int gD = 20;
			static int bD = 20;
			cv::createTrackbar("rD", "treated", &rD, 100);
			cv::createTrackbar("gD", "treated", &gD, 100);
			cv::createTrackbar("bD", "treated", &bD, 100);

			static int rO = 100;
			static int gO = 100;
			static int bO = 100;
			cv::createTrackbar("rD", "segmented", &rO, 200);
			cv::createTrackbar("gD", "segmented", &gO, 200);
			cv::createTrackbar("bD", "segmented", &bO, 200);

			int r, g, b;
			int rVec, gVec, bVec;

			const int size = input.rows * input.cols * 3;
			for (auto i = 0; i < size; i += 3)
			{
				b = input.data[i];
				g = input.data[i + 1];
				r = input.data[i + 2];

				bVec = b - bC;
				gVec = g - gC;
				rVec = r - rC;

				b = bC + bVec * (float(bD) / 20);
				g = gC + gVec * (float(gD) / 20);
				r = rC + rVec * (float(rD) / 20);

				output.data[i] = min(max(b + bO - 100, 0), 255);
				output.data[i + 1] = min(max(g + gO - 100, 0), 255);
				output.data[i + 2] = min(max(r + rO - 100, 0), 255);
			}
		}

		/*void track(cv::Mat& frame)
		{
		cv::imshow("frame", frame);

		cv::Mat treated = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		treat(frame, treated);
		cv::imshow("treated", treated);

		cv::Mat adapted = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		adapt(treated, adapted);
		cv::imshow("adapted", adapted);

		cv::Mat segmented = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		Segmenter::segment(adapted, segmented, lut);
		cv::imshow("segmented", segmented);
		}*/

	private:
		

		BGRLumminanceTransformer(){
			//HSVSobottkaPitas classifier;
			//BayesianPixelClassifier classifier(0.05f, CV_BGR, true, true, true);
			//SingleGaussianClassifier classifier(0.05f, CV_BGR, false, true, true);      
			//LUTManager::saveLUT(classifier, "../../../tools/HAFT/rsc/skin luts/BayesianPixelClassifier.64.lut", 64);
			/*LUTManager::loadLUT("../../Resource/Skin Luts/HSVSobottkaPitas.64.lut", lut);*/
		};            // Constructor? (the {} brackets) are needed here.
		// Dont forget to declare these two. You want to make sure they
		// are unaccessable otherwise you may accidently get copies of
		// your singleton appearing.
		BGRLumminanceTransformer(BGRLumminanceTransformer const&);       // Don't Implement
		void operator=(BGRLumminanceTransformer const&); // Don't implement
	};
}