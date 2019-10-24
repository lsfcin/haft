/*
* This file is subject to the terms and conditions defined in
* file 'LICENSE.txt', which is part of this source code package.
*/

#include "AdaboostDetector.h"

#include "Globals.h"

haft::AdaboostDetector& haft::AdaboostDetector::instance()
{
	static AdaboostDetector _instance; // Guaranteed to be destroyed.
	// Instantiated on first use.
	return _instance;
}

bool haft::AdaboostDetector::detect(const cv::Mat& grayImage, std::vector<cv::Rect>& targets, TargetType type, double scale)
{
	auto detected = false;

	cv::Mat smallImg(cvRound(grayImage.rows / scale), cvRound(grayImage.cols / scale), CV_8UC1);
	cv::resize(grayImage, smallImg, smallImg.size(), 0, 0, CV_INTER_LINEAR);
	cv::equalizeHist(smallImg, smallImg);

	if (type == HAND) cascadeHand.detectMultiScale(smallImg, targets, 1.2, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(40, 40));
	if (type == FACE) cascadeFace.detectMultiScale(smallImg, targets, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
	if (type == BODY) cascadeBody.detectMultiScale(smallImg, targets, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

	//|CV_HAAR_DO_ROUGH_SEARCH
	//|CV_HAAR_DO_CANNY_PRUNING
	//|CV_HAAR_SCALE_IMAGE

	return detected;
}

haft::AdaboostDetector::AdaboostDetector()
{
	if (!cascadeFace.load(haft::Globals::resourcesRootUrl + "/Adaboost XMLs/haarcascades/haarcascade_frontalface_alt.xml") ||
		!cascadeHand.load(haft::Globals::resourcesRootUrl + "/Adaboost XMLs/haarcascades/haarcascade_closedhand.xml") ||
		!cascadeBody.load(haft::Globals::resourcesRootUrl + "/Adaboost XMLs/haarcascades/haarcascade_upperbody.xml"))
	{
		std::cerr
			<< "ERROR: Could not load classifier cascade." << std::endl
			<< "Please check if the training files are in the correct path: " << std::endl
			<< haft::Globals::resourcesRootUrl + "/Adaboost XMLs/haarcascades/" << std::endl;

		std::cerr << 
			"Usage: facedetect [--cascade=<cascade_path>]\n"
			"                  [--nested-cascade[=nested_cascade_path]]\n"
			"                  [--scale[=<image scale>\n"
			"                  [filename|camera_index]\n" << std::endl;
		exit(0);
	}
}
