#include "voxar_haft.h"

void haft::track(cv::Mat& frame, std::vector<Target>& targets)
{
	Tracker::getInstance().track(frame);
	targets = Tracker::getInstance().targets;
}

void haft::trackFromCamera(int camID, std::vector<Target>& targets)
{
	static cv::VideoCapture capture;
	static auto opened = false;

	if (!opened)
	{
		capture.open(camID);
		capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

		if (!capture.isOpened())
		{
			std::cerr << "ERROR: could not initialize the camera\n" << std::endl;
			system("pause");
			exit(0);
		}
		else
		{
			opened = true;
		}
	}

	static cv::Mat frame;
	capture.read(frame);
	cv::flip(frame, frame, 1);

	if (frame.data != nullptr)
		track(frame, targets);
}

void haft::trackFromVideo(const std::string& videoURL, std::vector<Target>& targets)
{
	static cv::VideoCapture capture;
	static auto opened = false;

	if (!opened)
	{
		capture.open(videoURL);

		if (!capture.isOpened())
		{
			std::cerr << "ERROR: could not load the video\n" << std::endl;
			system("pause");
			exit(0);
		}
		else
		{
			opened = true;
		}
	}

	static cv::Mat frame;
	capture.read(frame);

	// if frame is empty then the video reached its end, restart the capture
	if (frame.empty())
	{
		opened = false;
	}
	// otherwise, keep tracking
	else
	{
		track(frame, targets);
	}
}

void haft::trackFromImage(const std::string& imageURL, std::vector<Target>& targets)
{
	static cv::VideoCapture capture;
	static auto opened = false;

	static cv::Mat frame;
	frame = cv::imread(imageURL);

	if (!(frame.data))
	{
		std::cerr << "ERROR: could not load image\n" << std::endl;
		system("pause");
		exit(0);
	}

	track(frame, targets);
}