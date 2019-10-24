/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2\opencv.hpp>

#include <vector>
#include <string>
#include <stdio.h>
#include <fstream>

namespace haft
{	
	#define PI 3.14159265
	#define RADIANS_TO_DEGREES 57.2957795
	#define CV_BGR -1

	enum PRINT_MODE{ PRINT_NOTHING, PRINT_DELTA, PRINT_MEAN };

	namespace Timer
	{
		static double time = 0;
		static double sum = 0;
		static double count = 0;

		static void start()
		{
			time = double(cv::getTickCount());
		}

		static double print(PRINT_MODE mode = PRINT_DELTA, std::string prefix = "time: ")
		{
			const double delta = (double(cv::getTickCount()) - time) * 1000.0 / cv::getTickFrequency();
			if (mode == PRINT_MEAN)
			{
				sum += delta;
				++count;
				std::cout << prefix << sum / count << std::endl;
			}
			else if (mode == PRINT_DELTA)
			{
				std::cout << prefix << delta << std::endl;
			}
			return delta;
		}
	};

	template <typename T>
	class Stack
	{
	public:
		Stack(const unsigned int& max) : max(max), end(0)
		{
			stack = new T[max];
		}

		~Stack()
		{
			delete[] stack;
		}

		void push(T& element)
		{
			stack[end] = element;
			++end;
		}

		T& pop()
		{
			return stack[--end];
		}

		T& top()
		{
			return stack[end - 1];
		}

		bool empty()
		{
			return end == 0;
		}

	private:
		unsigned int max;
		unsigned int end;
		T* stack;
	};

	template <typename T>
	class Queue
	{
	public:
		Queue(const unsigned int& max) : max(max), start(0), end(0)
		{
			queue = new T[max];
		}

		~Queue()
		{
			delete[] queue;
		}

		void push(T element)
		{
			queue[end] = element;
			++end;
		}

		T& pop()
		{
			return queue[start++];
		}

		T& front()
		{
			return queue[start];
		}

		bool empty()
		{
			return (end - start) == 0;
		}

	private:
		unsigned int max;
		unsigned int start;
		unsigned int end;
		T* queue;
	};


	class Group
	{
	public:
		Group(int start, int end, int max) : start(start), end(end), max(max) {}
		int start; //......//id of the first element
		int end; //........//id of the last element
		int max; //........//number of elements of the vector that generated this group

		unsigned int size()
		{
			int temp = end;
			if (temp < start) temp += max;
			return temp - start + 1;
		}
	};

	//Util can't be instantiated, it's just a holder for static functions
	class Util
	{
	public:

		enum PolygonType {CONVEX, CONCAVE};

		inline static std::vector<cv::Point> scalePolygon(
			const std::vector<cv::Point>& polygon,
			float scale)
		{
			std::vector<cv::Point> result;

			// first find the centroid
			auto centroid = cv::Point(0, 0);
			for each (auto &point in polygon)
			{
				centroid += point;
			}
			centroid.x /= polygon.size();
			centroid.y /= polygon.size();

			// now describe all points according to the centroid
			// then scale each point
			// and finally transform then back to their original coordinate system
			for each (auto &point in polygon)
			{
				auto temp = point - centroid;
				temp.x *= scale;
				temp.y *= scale;
				temp += centroid;

				result.push_back(temp);
			}

			return result;
		}

		inline static void drawPolygon(
			const std::vector<cv::Point>& polygon,
			const cv::Point& offset,
			cv::Mat& mask,
			float scale,
			cv::Scalar color,
			int thickness = 1)
		{
			auto scaled = scalePolygon(polygon, scale);

			for (auto &point : scaled)
			{
				point.x += offset.x;
				point.y += offset.y;
				if (point.x < 0)  point.x = 0;
				if (point.y < 0)  point.y = 0;
				if (point.x >= mask.cols)  point.x = mask.cols - 1;
				if (point.y >= mask.rows)  point.y = mask.rows - 1;
			}

			for (auto &point : scaled)
			{
				cv::line(mask, point, point, color, thickness);
			}
		}

		inline static void fillPolygon(
			const std::vector<cv::Point>& polygon,
			const cv::Point& offset,
			cv::Mat& mask,
			float scale,
			cv::Scalar color,
			PolygonType type = PolygonType::CONVEX)
		{
			auto scaled = scalePolygon(polygon, scale);

			for (auto &point : scaled)
			{
				point.x += offset.x;
				point.y += offset.y;
				if (point.x < 0)  point.x = 0;
				if (point.y < 0)  point.y = 0;
				if (point.x >= mask.cols)  point.x = mask.cols - 1;
				if (point.y >= mask.rows)  point.y = mask.rows - 1;
			}

			if (type == CONVEX)
			{
				cv::fillConvexPoly(mask, scaled, color);
			}
			else if (type == CONCAVE)
			{
				std::vector<cv::Point> tmp = scaled;
				const cv::Point* elementPoints[1] = { &tmp[0] };
				int numberOfPoints = (int)tmp.size();

				cv::fillPoly(mask, elementPoints, &numberOfPoints, 1, color, 8);
			}
		}

		inline static void load(const std::string& url, std::ifstream &stream)
		{
			if (std::strcmp(url.c_str(), "") != 0)
			{
				stream.open(url);
				if (!stream.is_open())
				{
					std::cout << "ERROR: file not loaded: " << url << std::endl;
					system("pause");
					exit(0);
				}
			}
		}

		inline static float percentageOf(const float& value1, const float& value2)
		{
			float percent1 = value1 / value2;
			float percent2 = value2 / value1;
			float result = percent1;
			if (result > percent2) result = percent2;
			return result;
		}

		inline static bool  equals(const float& a, const float& b)
		{
			return abs(a - b) < 0.000001f;
		}
		inline static bool  equals(const double& a, const double& b)
		{
			return abs(a - b) < 0.00000000001;
		}
		template <typename T = float> inline static float dir(const cv::Point_<T>& a, const cv::Point_<T>& b, const cv::Point_<T>& c)
		{
			const cv::Point_<T> vec1(b.x - a.x, b.y - a.y);
			const cv::Point_<T> vec2(b.x - c.x, b.y - c.y);

			return dir(vec1, vec2);
		}
		template <typename T = float> inline static float dir(const cv::Point_<T>& vec1, const cv::Point_<T>& vec2)
		{
			return (vec1.x * vec2.y) - (vec1.y * vec2.x);
		}
		template <typename T = float> inline static float dot(const cv::Point_<T>& v1, const cv::Point_<T>& v2)
		{
			return v1.x * v2.x + v1.y * v2.y;
		}
		template <typename T = float> inline static float cos(const cv::Point_<T>& a, const cv::Point_<T>& b, const cv::Point_<T>& c)
		{
			const cv::Point_<T> vec1(b.x - a.x, b.y - a.y);
			const cv::Point_<T> vec2(b.x - c.x, b.y - c.y);

			const float cosine = cos(vec1, vec2);

			return cosine;
		}
		template <typename T = float> inline static float cos(const cv::Point_<T>& vec1, const cv::Point_<T>& vec2)
		{
			const float cosine = dot(vec1, vec2) / (norm(vec1) * norm(vec2));

			return cosine;
		}
		template <typename T = float> inline static float orientation(const cv::Point_<T>& direction)
		{
			double cosine = Util::cos(direction, cv::Point_<T>(0, 0), cv::Point_<T>(1, 0));
			double sine = Util::cos(direction, cv::Point_<T>(0, 0), cv::Point_<T>(0, 1));

			double orientationCosine1 = acos(cosine) * RADIANS_TO_DEGREES;
			double orientationSine1 = acos(sine) * RADIANS_TO_DEGREES;
			double orientationCosine2 = 360 - orientationCosine1;
			double orientationSine2 = 360 - orientationCosine1;

			double orientation = 0;
			if (equals(orientationCosine1, orientationSine1)) orientation = orientationCosine1;
			else if (equals(orientationCosine1, orientationSine2)) orientation = orientationCosine1;
			else if (equals(orientationCosine2, orientationSine1)) orientation = orientationCosine2;
			else if (equals(orientationCosine2, orientationSine2)) orientation = orientationCosine2;

			return orientation;
		}
		inline static int perfectPowerOf2Multiplier(unsigned int numerator, unsigned int denominator)
		{
			int result = 0;

			unsigned int temp;
			unsigned int rest = 0;
			while (numerator > denominator && rest == 0)
			{
				temp = numerator >> 1;
				rest = (temp << 1) - numerator;
				numerator = temp;
				result++;
			}

			if (rest != 0) result = -1;

			return result;
		}
		inline static unsigned int nonBlackCount(const cv::Mat& image)
		{
			unsigned int result = 0;
			const unsigned int size = image.cols * image.rows * image.channels();

			for (unsigned int i = 0; i < size; ++i)
			{
				if (image.data[i] != 0)
				{
					++result;
				}
			}

			return result;
		}
		template <typename T = float> inline static cv::Point_<T> midPoint(const cv::Rect& rect)
		{
			return cv::Point_<T>(rect.x + (rect.width >> 1), rect.y + (rect.height >> 1));
		}
		template <typename T = float> inline static cv::Point_<T> midPoint(const std::vector<unsigned int>& pixelsIDs, const cv::Size& imageSize)
		{
			unsigned int sumX = 0;
			unsigned int sumY = 0;
			const unsigned int vectorSize = pixelsIDs.size();
			cv::Point_<T> point;

			for (unsigned int i = 0; i < vectorSize; ++i)
			{
				point = Util::index2D(pixelsIDs[i], imageSize);
				sumX += point.x;
				sumY += point.y;
			}

			auto x = sumX / vectorSize;
			auto y = sumY / vectorSize;

			return cv::Point_<T>(x, y);
		}
		template <typename T1 = float, typename T2 = float> inline static cv::Point_<T1> midPoint(const std::vector<cv::Point_<T2>>& points)
		{
			unsigned int sumX = 0;
			unsigned int sumY = 0;
			const unsigned int vectorSize = points.size();

			for (unsigned int i = 0; i < vectorSize; ++i)
			{
				sumX += points[i].x;
				sumY += points[i].y;
			}

			auto x = sumX / vectorSize;
			auto y = sumY / vectorSize;

			return cv::Point_<T1>(x, y);
		}
		template <typename T = float> inline static cv::Point_<T> midPoint(const cv::Point_<T>& p1, const cv::Point_<T>& p2)
		{
			cv::Point_<T> point;
			point.x = (p1.x + p2.x) / 2;
			point.y = (p1.y + p2.y) / 2;
			return point;
		}
		inline static cv::Rect pointsROI(const std::vector<unsigned int>& pixelsIDs, const cv::Size& imageSize)
		{
			unsigned int minX = INT_MAX;
			unsigned int maxX = 0;
			unsigned int minY = INT_MAX;
			unsigned int maxY = 0;
			cv::Point2f point;

			const unsigned int vectorSize = pixelsIDs.size();

			for (unsigned int i = 0; i < vectorSize; ++i)
			{
				point = Util::index2D(pixelsIDs[i], imageSize);
				if (point.x < minX) minX = point.x;
				if (point.x > maxX) maxX = point.x;
				if (point.y < minY) minY = point.y;
				if (point.y > maxY) maxY = point.y;
			}

			return cv::Rect(cv::Point2f(minX, minY), cv::Point2f(maxX, maxY));
		}
		template <typename T = float> inline static unsigned int distance4C(const cv::Point_<T>& p1, const cv::Point_<T>& p2) //4-connectivity distance, is used only due its speed.
		{
			return abs(p1.x - p2.x) + abs(p1.y - p2.y);
		}
		template <typename T = float> inline static float distanceEuclidean(const cv::Point_<T>& p1, const cv::Point_<T>& p2)
		{
			return sqrt(pow((double)(p1.x - p2.x), 2) + pow((double)(p1.y - p2.y), 2));
		}
		inline static std::string itos(int number)
		{
			std::stringstream ss;//create a stringstream
			ss << number;//add number to the stream
			return ss.str();//return a string with the contents of the stream
		}
		template <typename T = float> inline static bool isInside(const cv::Mat& image, const cv::Point_<T>& point)
		{
			return point.x >= 0 && point.y >= 0 && point.x < image.cols && point.y < image.rows;
		}
		template <typename T = float> inline static bool isInside(const cv::Rect& rect, const cv::Point_<T>& point)
		{
			return point.x >= rect.x && point.y >= rect.y && point.x < rect.x + rect.width && point.y < rect.y + rect.height;
		}
		inline static bool isInside(int x, int y, int w, int h)
		{
			return x >= 0 && y >= 0 && x < w && y < h;
		}
		template <typename T = float> inline static bool isValid(const cv::Mat& image, const cv::Point_<T>& point)
		{
			return isInside(image, point) && image.at<bool>(point);
		}
		template <typename T = float> inline static void boundingBox(
			const std::vector<cv::Point_<T>>& points, 
			const cv::Size& imageSize, 
			float sizeIncrement, 
			cv::Rect& box)
		{ //TODO usar o desvio padrão das relevâncias talvez, usar region growing no currP2D talvez   
			box = cv::boundingRect(cv::Mat(points));
			box.x -= sizeIncrement;
			box.y -= sizeIncrement;
			box.width += sizeIncrement * 2;
			box.height += sizeIncrement * 2;
			limitBox(imageSize, box);
		}
		template <typename T = float> inline static void commonBox(
			const cv::Point_<T>& center, 
			const cv::Size& imageSize, 
			float sizeRelativeIncrement, 
			cv::Rect& box)
		{
			const unsigned int sizeRealIncrement = sizeRelativeIncrement * imageSize.width;
			box.x = center.x - sizeRealIncrement;
			box.y = center.y - sizeRealIncrement;
			box.width = 2 * sizeRealIncrement;
			box.height = 2 * sizeRealIncrement;
			limitBox(imageSize, box);
		}

		inline static void getHistogram1DImg(const cv::Mat& histogram, cv::Mat& image)
		{
			cv::Mat normalizedHist;
			const unsigned int binW = 2;
			cv::normalize(histogram, normalizedHist, 0, 255, CV_MINMAX);
			image = cv::Mat::zeros(200, normalizedHist.rows * binW, CV_8UC3);
			cv::Mat buf(1, normalizedHist.size, CV_8UC3);
			for (unsigned int i = 0; i < normalizedHist.rows; i++)
				buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / normalizedHist.rows), 255, 255);

			cv::cvtColor(buf, buf, CV_HSV2BGR);

			for (unsigned int i = 0; i < normalizedHist.rows; i++)
			{
				const unsigned int colHeight = cv::saturate_cast<int>(normalizedHist.at<float>(i) * image.rows / 255);
				rectangle(image,
					cv::Point(i * binW, image.rows),
					cv::Point((i + 1) * binW, image.rows - colHeight),
					cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
			}
		}
		
		inline static void getHistogram2DImg(const cv::Mat& histogram, 
			cv::Mat& histImg,
			const unsigned int scale = 4)
		{
			double maxVal = 0;
			cv::minMaxLoc(histogram, 0, &maxVal, 0, 0);

			const unsigned int bins1 = histogram.cols;
			const unsigned int bins2 = histogram.rows;
			histImg = cv::Mat::zeros(bins1*scale, bins2*scale, CV_8UC3);

			for (unsigned int b2 = 0; b2 < bins2; b2++)
			{
				for (unsigned int b1 = 0; b1 < bins1; b1++)
				{
					float binVal = histogram.at<float>(b2, b1);
					unsigned int intensity = cvRound(binVal * 255 / maxVal);
					cv::rectangle(histImg, cv::Point(b2*scale, b1*scale),
						cv::Point((b2 + 1)*scale - 1, (b1 + 1)*scale - 1),
						cv::Scalar::all(intensity),
						CV_FILLED);
				}
			}
		}
		
		// TODO: complete this method
		inline static void createHistogramImg(cv::Mat& histImg, 
			cv::Mat& finalImg, 
			int conversion)
		{
			cv::resize(histImg, histImg, cv::Size(255, 255));

			const unsigned int rows = histImg.rows;
			const unsigned int cols = histImg.cols;

			finalImg = cv::Mat::zeros(rows, cols, CV_8UC3);

			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					if (conversion == CV_BGR2YCrCb)
					{
						finalImg.at<cv::Vec3b>(i, j)[0] = 128;
						finalImg.at<cv::Vec3b>(i, j)[2] = i;
						finalImg.at<cv::Vec3b>(i, j)[1] = j;
					}
					else if (conversion == CV_BGR)
					{
						finalImg.at<cv::Vec3b>(i, j)[0] *= histImg.at<cv::Vec3b>(i, j)[0];
						finalImg.at<cv::Vec3b>(i, j)[1] *= histImg.at<cv::Vec3b>(i, j)[1];
						finalImg.at<cv::Vec3b>(i, j)[2] *= histImg.at<cv::Vec3b>(i, j)[2];
					}
					else if (conversion == CV_BGR2HSV)
					{

					}
				}
			}

			if (conversion == CV_BGR2YCrCb)
				cv::cvtColor(finalImg, finalImg, CV_YCrCb2BGR);
			else if (conversion == CV_BGR2HSV)
				cv::cvtColor(finalImg, finalImg, CV_HSV2BGR);

			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					finalImg.at<cv::Vec3b>(i, j)[0] = (char)(finalImg.at<cv::Vec3b>(i, j)[0] * (float(histImg.at<cv::Vec3b>(i, j)[0]) / 255));
					finalImg.at<cv::Vec3b>(i, j)[1] = (char)(finalImg.at<cv::Vec3b>(i, j)[1] * (float(histImg.at<cv::Vec3b>(i, j)[1]) / 255));
					finalImg.at<cv::Vec3b>(i, j)[2] = (char)(finalImg.at<cv::Vec3b>(i, j)[2] * (float(histImg.at<cv::Vec3b>(i, j)[2]) / 255));
				}
			}
		}

		inline static float smooth(const std::vector<float>& data)
		{
			const unsigned int dataSize = data.size();
			float ratio = 0;
			for (int i = 1; i < dataSize; ++i)
			{
				ratio += std::max(data[i] / data[i - 1], data[i - 1] / data[i]);
			}
			ratio /= dataSize - 1;

			float smoothed = data[0];

			for (int i = 1; i < dataSize; ++i)
			{
				if (std::max(smoothed / data[i - 1], data[i - 1] / smoothed) <= ratio)
					smoothed = data[i];
				else
					if (smoothed < data[i - 1])
						smoothed *= ratio;
					else
						smoothed /= ratio;
			}

			return smoothed;
		}
		inline static cv::Rect intersection(const cv::Rect& rect1, const cv::Rect& rect2)
		{
			const int x1 = std::max(rect1.x, rect2.x);
			const int y1 = std::max(rect1.y, rect2.y);
			const int x2 = std::min(rect1.br().x, rect2.br().x);
			const int y2 = std::min(rect1.br().y, rect2.br().y);

			return cv::Rect(x1, y1, x2 - x1, y2 - y1);
		}
		inline static float intersectionPercentage(const cv::Rect& rect1, const cv::Rect& rect2, bool high = true)
		{ //bool high tells if the percentage wanted is the higher one or the lower
			cv::Rect intersecRect = intersection(rect1, rect2);
			float percentage = 0;
			if (intersecRect.width > 0 && intersecRect.height > 0)
			{
				if (high) percentage = std::max((double)intersecRect.area() / rect1.area(), (double)intersecRect.area() / rect2.area());
				else     percentage = std::min((double)intersecRect.area() / rect1.area(), (double)intersecRect.area() / rect2.area());
			}
			return percentage;
		}
		template <typename T = float> inline static unsigned int index1D(const cv::Point_<T>& point, const cv::Size& imageSize)
		{
			return cvRound(point.x) + (cvRound(point.y) * imageSize.width);
		}
		inline static unsigned int index1D(const unsigned int& x, const unsigned int& y, const unsigned int& width)
		{
			return x + (y * width);
		}
		inline static unsigned int index1D(const unsigned int& x,
			const unsigned int& y,
			const unsigned int& z,
			const unsigned int& width,
			const unsigned int& height)
		{
			return x + (y * width) + (z * width * height);
		}
		template <typename T = float> inline static cv::Point_<T> index2D(unsigned int index, const cv::Size& imageSize)
		{
			return cv::Point_<T>(index % imageSize.width, index / imageSize.width);
		}
		inline static void limitBox(const cv::Size& imageSize, cv::Rect& box)
		{
			if (box.x < 0) box.x = 0;
			if (box.y < 0) box.y = 0;
			if (box.width >= imageSize.width)  box.width = imageSize.width - 1;
			if (box.height >= imageSize.height) box.height = imageSize.height - 1;
			if (box.x + box.width >= imageSize.width)  box.x = imageSize.width - box.width - 1;
			if (box.y + box.height >= imageSize.height) box.y = imageSize.height - box.height - 1;
		}
		template <typename T = float> inline static float norm(const cv::Point_<T>& p) {
			return sqrt(pow(p.x, 2) + pow(p.y, 2));
		}
		template <typename T = float> inline static bool isNearOther(
			unsigned int i, 
			unsigned int minDistance, 
			unsigned int& other,
			const unsigned int& pointsCount, 
			const std::vector<cv::Point_<T>>& points)
		{
			bool isNear = false;
			for (unsigned int j = i + 1; j < pointsCount; ++j) {
				if (distanceEuclidean(points[i], points[j]) < minDistance) {
					other = j;
					isNear = true;
					break;
				}
			}
			return isNear;
		}
		template <typename T = float> inline static cv::Point_<T> multiply(const cv::Point_<T>& point, float factor)
		{
			return cv::Point_<T>(point.x * factor, point.y * factor);
		}
		template <typename T = float> inline static float angleBetween(const cv::Point_<T> &v1, const cv::Point_<T> &v2)
		{
			T atanA = atan2(v1.x, v1.y);
			T atanB = atan2(v2.x, v2.y);

			return atanA - atanB;
		}
		template <typename T = float> inline static cv::Point_<T> rotate(const cv::Point_<T> &vec, float angleRad)
		{
			auto result = cv::Point_<T>();

			float sine = std::sin(angleRad);
			float cosine = std::cos(angleRad);

			result.x = vec.x * cosine - vec.y * sine;
			result.y = vec.x * sine + vec.y * cosine;

			return result;
		}
		template <typename T = float> inline static cv::Point_<T> mirror(const cv::Point_<T> &vecSource, const cv::Point_<T> &vecMirror)
		{
			auto reflected = cv::Point_<T>();

			auto angle = angleBetween(vecSource, vecMirror);

			reflected = rotate(vecSource, angle * 2);

			return reflected;
		}
		template <typename T = float> inline static cv::Point_<T> subtractPoints(const cv::Point_<T>& point1, const cv::Point_<T>& point2)
		{
			return cv::Point_<T>(point1.x - point2.x, point1.y - point2.y);
		}
		template <typename T = float> inline static float getNorm(const cv::Point_<T>& point)
		{
			return sqrt(point.x * point.x + point.y * point.y);
		}
		template <typename T = float> inline static void normalize(cv::Point_<T>& point)
		{
			unsigned int norm = getNorm(point);

			point.x /= norm;
			point.y /= norm;
		}
		template <typename T = float> inline static void setNorm(cv::Point_<T>& point, float newNorm) {
			normalize(point);

			point.x *= newNorm;
			point.y *= newNorm;
		}
		template <typename T = float> inline static cv::Point_<T> findPoint(
			const cv::Point_<T>& point1, 
			const cv::Point_<T>& point2, 
			float t)
		{
			cv::Point_<T> result;

			result = subtractPoints(point1, point2);
			setNorm(result, getNorm(result) * t);

			result.x = point1.x - result.x;
			result.y = point1.y - result.y;

			return result;
		}
		template <typename T = float> inline static float intersect(cv::Point_<T>& a, cv::Point_<T>& b, cv::Point_<T>& c, cv::Point_<T>& d)
		{ /** /
		  E = B-A = ( Bx-Ax, By-Ay )
		  F = D-C = ( Dx-Cx, Dy-Cy )
		  P = ( -Ey, Ex )
		  h = ( (A-C) * P ) / ( F * P )
		  //*/

			float result = -1;

			float det = (d.x - c.x) * (b.y - a.y) - (d.y - c.y) * (b.x - a.x);

			if (det != 0)
			{
				float s = ((d.x - c.x) * (c.y - a.y) - (d.y - c.y) * (c.x - a.x)) / det;

				if (s > 0 && s < 1)
				{
					result = s;
				}
			}

			return result;
		}
		template <typename T = float> inline static void followPoint(
			cv::Point_<T> &follower, 
			cv::Point_<T> &followed, 
			float velocity, 
			unsigned int jitterReductionTerm)
		{
			if (followed.x >= 0 && followed.y >= 0)
			{
				if (follower.x < 0 || follower.y < 0)
				{
					follower = followed;
				}
				else
				{
					cv::Point_<T> distance = subtractPoints(follower, followed);

					if (abs(distance.x) > jitterReductionTerm ||
						abs(distance.y) > jitterReductionTerm)
					{
						cv::Point_<T> increment = multiply(distance, velocity);

						follower.x -= increment.x;
						follower.y -= increment.y;
					}
				}
			}
		}
		inline static unsigned int calcPowerOf2(unsigned int n)
		{
			unsigned int powerOf2 = 2;
			while (n > 1)
			{
				n = n / 2;
				powerOf2 = powerOf2 * 2;
			}
			return powerOf2;
		}
		template <typename T = float> inline static std::vector<unsigned int> orientationsHistogram(
			std::vector<cv::Point_<T>> directions, 
			int numBins = 8)
		{
			std::vector<unsigned int> histogram(numBins, 0);

			const unsigned int divisor = 360 / numBins;
			const unsigned int size = directions.size();

			unsigned int ori;

			for (int i = 0; i < size; ++i)
			{
				ori = orientation(directions[i]);
				histogram[ori / divisor]++;
			}
			return histogram;
		}

	private:
		Util(){}
		~Util(){}
		Util(Util const&);            //hiding the copy constructor
		Util& operator=(Util const&); //hiding the assignment operator
	};
}