/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace haft
{
#define PI 3.1415926535897932384626433832795
#define EPSILON 1e-14

	int findNearestPoint(const cv::Point2f &point, std::vector<cv::Point2f> &points)
	{
		double minDistance = DBL_MAX;
		int minIdx = -1;

		int numPoints = points.size();

		for (int i = 0; i < numPoints; ++i)
		{
			double distance = cv::norm(point - points[i]);
			if (distance < minDistance)
			{
				minDistance = distance;
				minIdx = i;
			}
		}

		return minIdx;
	}

	unsigned char computeG1(unsigned char x1, unsigned char x2, unsigned char x3, unsigned char x4,
		unsigned char x5, unsigned char x6, unsigned char x7, unsigned char x8)
	{
		unsigned char b1 = (~x1 & (x2 | x3)) & 0x01;
		unsigned char b2 = (~x3 & (x4 | x5)) & 0x01;
		unsigned char b3 = (~x5 & (x6 | x7)) & 0x01;
		unsigned char b4 = (~x7 & (x8 | x1)) & 0x01;

		return ((b1 + b2 + b3 + b4) == 1 ? 0xFF : 0);
	}

	unsigned char computeG2(unsigned char x1, unsigned char x2, unsigned char x3, unsigned char x4,
		unsigned char x5, unsigned char x6, unsigned char x7, unsigned char x8)
	{
		unsigned char n1 = ((x1 | x2) & 0x01) + ((x3 | x4) & 0x01) + ((x5 | x6) & 0x01) + ((x7 | x8) & 0x01);
		unsigned char n2 = ((x2 | x3) & 0x01) + ((x4 | x5) & 0x01) + ((x6 | x7) & 0x01) + ((x8 | x1) & 0x01);

		unsigned char n1n2min = std::min(n1, n2);

		return (n1n2min >= 2 && n1n2min <= 3 ? 0xFF : 0);
	}

	unsigned char computeG3(unsigned char x1, unsigned char x2, unsigned char x3, unsigned char x8)
	{
		return ~((x2 | x3 | ~x8) & x1);
	}

	unsigned char computeG3prime(unsigned char x4, unsigned char x5, unsigned char x6, unsigned char x7)
	{
		return ~((x6 | x7 | ~x4) & x5);
	}

	void thin(cv::Mat &src, cv::Mat &dst)
	{
		int srcType = src.type();
		int dstType = dst.type();

		cv::Size srcSize = src.size();
		cv::Size dstSize = dst.size();

		assert(srcType == CV_8UC1);
		assert(srcType == dstType);
		assert(srcSize == dstSize);

		int cols = srcSize.width - 1;
		int rows = srcSize.height - 1;

		// first iteration
		cv::Mat it1(srcSize, srcType, cv::Scalar());

		for (int i = 1; i < rows; ++i)
		{
			unsigned char *it1Ptr = it1.ptr<unsigned char>(i);
			unsigned char *iPtr = src.ptr<unsigned char>(i);
			unsigned char *northPtr = src.ptr<unsigned char>(i - 1);
			unsigned char *southPtr = src.ptr<unsigned char>(i + 1);

			for (int j = 1; j < cols; ++j)
			{
				unsigned char cur = iPtr[j];

				if (cur == 0xFF)
				{
					int east = j + 1;
					int west = j - 1;

					unsigned char x1 = iPtr[east];
					unsigned char x2 = northPtr[east];
					unsigned char x3 = northPtr[j];
					unsigned char x4 = northPtr[west];
					unsigned char x5 = iPtr[west];
					unsigned char x6 = southPtr[west];
					unsigned char x7 = southPtr[j];
					unsigned char x8 = southPtr[east];

					unsigned char g1 = computeG1(x1, x2, x3, x4, x5, x6, x7, x8);
					unsigned char g2 = computeG2(x1, x2, x3, x4, x5, x6, x7, x8);
					unsigned char g3 = computeG3(x1, x2, x3, x8);

					it1Ptr[j] = ~(g1 & g2 & g3);
				}
			}
		}

		// clearing borders
		for (int i = 0; i < srcSize.height; ++i)
		{
			unsigned char *dstPtr = dst.ptr<unsigned char>(i);
			dstPtr[0] = dstPtr[cols] = 0;
		}

		unsigned char *dstPtr1 = dst.ptr<unsigned char>(0);
		unsigned char *dstPtr2 = dst.ptr<unsigned char>(rows);

		for (int i = 0; i < srcSize.width; ++i)
		{
			dstPtr1[i] = dstPtr2[i] = 0;
		}

		// second iteration
		for (int i = 1; i < rows; ++i)
		{
			unsigned char *it2Ptr = dst.ptr<unsigned char>(i);
			unsigned char *iPtr = it1.ptr<unsigned char>(i);
			unsigned char *northPtr = it1.ptr<unsigned char>(i - 1);
			unsigned char *southPtr = it1.ptr<unsigned char>(i + 1);

			for (int j = 1; j < cols; ++j)
			{
				it2Ptr[j] = iPtr[j];

				if (it2Ptr[j] == 0xFF)
				{
					int east = j + 1;
					int west = j - 1;

					unsigned char x1 = iPtr[east];
					unsigned char x2 = northPtr[east];
					unsigned char x3 = northPtr[j];
					unsigned char x4 = northPtr[west];
					unsigned char x5 = iPtr[west];
					unsigned char x6 = southPtr[west];
					unsigned char x7 = southPtr[j];
					unsigned char x8 = southPtr[east];

					unsigned char g1 = computeG1(x1, x2, x3, x4, x5, x6, x7, x8);
					unsigned char g2 = computeG2(x1, x2, x3, x4, x5, x6, x7, x8);
					unsigned char g3prime = computeG3prime(x4, x5, x6, x7);

					it2Ptr[j] = ~(g1 & g2 & g3prime);
				}
			}
		}
	}

	void canny(cv::Mat &src, cv::Mat &dst, double lowThresh, double highThresh, double sigma)
	{
		int srcType = src.type();
		int dstType = dst.type();

		cv::Size srcSize = src.size();
		cv::Size dstSize = dst.size();

		assert(srcType == CV_8UC1);
		assert(srcType == dstType);
		assert(srcSize == dstSize);

		// calculate derivatives
		int kernelSize = cvRound(sigma * 9) | 1;
		int range = kernelSize / 2;
		double ssq = sigma * sigma;

		cv::Mat dgau2D(kernelSize, kernelSize, CV_32F);

		for (int i = 0; i < kernelSize; ++i)
		{
			float *ptr = dgau2D.ptr<float>(i);

			for (int j = 0; j < kernelSize; ++j)
			{
				int y = i - range;
				int x = j - range;

				ptr[j] = (float)(-x * exp(-(x * x + y * y) / (2.0f * ssq)) / (PI * ssq));
			}
		}

		cv::Mat gau(1, kernelSize, CV_32F);

		float *ptr = gau.ptr<float>(0);

		for (int i = 0; i < kernelSize; ++i)
		{
			int x = i - range;
			ptr[i] = (float)(exp(-(x * x) / (2.0f * ssq)) / (2.0f * PI * ssq));
		}

		cv::Mat smooth(srcSize, CV_64F);

		/*cv::sepFilter2D(src, smooth, smooth.depth(), gau, gau.t());//*/
		cv::Mat scaled(srcSize, CV_64F);
		src.convertTo(scaled, CV_64F, 1.0 / 255.0);
		cv::sepFilter2D(scaled, smooth, smooth.depth(), gau, gau.t(), cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);//*/

		cv::Mat dx(srcSize, CV_64F);
		cv::Mat dy(srcSize, CV_64F);

		cv::filter2D(smooth, dx, dx.depth(), dgau2D, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
		cv::filter2D(smooth, dy, dy.depth(), dgau2D.t(), cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

		/*cv::Mat smooth2(srcSize, srcType);
		smooth.convertTo(smooth2, srcType, 255.0);
		cv::imwrite("smooth_opencv.bmp", smooth2);//*/

		dx *= -1;
		dy *= -1;

		/*cv::Mat dx2(srcSize, srcType);
		dx.convertTo(dx2, srcType, 255.0);
		cv::imwrite("dx_opencv.bmp", dx2);
		cv::Mat dy2(srcSize, srcType);
		dy.convertTo(dy2, srcType, 255.0);
		cv::imwrite("dy_opencv.bmp", dy2);//*/

		cv::Mat mag(srcSize, CV_64F);

		cv::sqrt(dx.mul(dx) + dy.mul(dy), mag);

		double magMaxVal = 0;
		cv::Point magMaxLoc;

		cv::minMaxLoc(mag, 0, &magMaxVal, 0, &magMaxLoc);

		//cout << "magMaxVal: " << magMaxVal << endl;
		//cout << "magMaxLoc: " << magMaxLoc << endl;

		if (magMaxVal > 0)
		{
			mag /= magMaxVal;
		}

		// non-maximum supression
		int rows = srcSize.height - 1;
		int cols = srcSize.width - 1;

		cv::Mat weakEdgeMap(srcSize, srcType, cv::Scalar());
		//cv::Mat strongEdgeMap(srcSize, srcType, cv::Scalar());

		std::vector<cv::Point> seeds;

		for (int i = 1; i < rows; ++i)
		{
			double *dxPtr = dx.ptr<double>(i);
			double *dyPtr = dy.ptr<double>(i);
			double *magPtr = mag.ptr<double>(i);
			double *northPtr = mag.ptr<double>(i - 1);
			double *southPtr = mag.ptr<double>(i + 1);

			unsigned char *weakPtr = weakEdgeMap.ptr<unsigned char>(i);
			//unsigned char *strongPtr = strongEdgeMap.ptr<unsigned char>(i);

			for (int j = 1; j < cols; ++j)
			{
				double dxVal = dxPtr[j];
				double dyVal = dyPtr[j];

				double gradMag = magPtr[j];

				double gradMag1 = 0;
				double gradMag2 = 0;

				int east = j + 1;
				int west = j - 1;

				if ((dyVal <= 0 && dxVal > -dyVal) || (dyVal >= 0 && dxVal < -dyVal))
				{
					double d = fabs(double(dyVal) / dxVal);
					gradMag1 = magPtr[east] * (1.0 - d) + northPtr[east] * d;
					gradMag2 = magPtr[west] * (1.0 - d) + southPtr[west] * d;
				}
				else if ((dxVal > 0 && -dyVal >= dxVal) || (dxVal < 0 && -dyVal <= dxVal))
				{
					double d = fabs(double(dxVal) / dyVal);
					gradMag1 = northPtr[j] * (1.0 - d) + northPtr[east] * d;
					gradMag2 = southPtr[j] * (1.0 - d) + southPtr[west] * d;
				}
				else if ((dxVal <= 0 && dxVal > dyVal) || (dxVal >= 0 && dxVal < dyVal))
				{
					double d = fabs(double(dxVal) / dyVal);
					gradMag1 = northPtr[j] * (1.0 - d) + northPtr[west] * d;
					gradMag2 = southPtr[j] * (1.0 - d) + southPtr[east] * d;
				}
				else if ((dyVal < 0 && dxVal <= dyVal) || (dyVal > 0 && dxVal >= dyVal))
				{
					double d = fabs(double(dyVal) / dxVal);
					gradMag1 = magPtr[west] * (1.0 - d) + northPtr[west] * d;
					gradMag2 = magPtr[east] * (1.0 - d) + southPtr[east] * d;
				}

				if (gradMag >= gradMag1 && gradMag >= gradMag2 && gradMag > lowThresh)
				{
					weakPtr[j] = 255;

					if (gradMag > highThresh)
					{
						cv::Point curPoint = cv::Point(j, i);
						seeds.push_back(curPoint);

						//strongPtr[j] = 255;
						//cout << curPoint << endl;
					}
				}
			}
		}

		//cv::imwrite("weak_edge_map_opencv.bmp", weakEdgeMap);
		//cv::imwrite("strong_edge_map_opencv.bmp", strongEdgeMap);

		// hysteresis
		cv::Mat fill(srcSize, srcType);

		cv::bitwise_not(weakEdgeMap, fill);

		cv::Mat mask(srcSize.height + 2, srcSize.width + 2, srcType, cv::Scalar(0));

		fill.copyTo(mask(cv::Rect(1, 1, srcSize.width, srcSize.height)));

		//cv::imwrite("mask.bmp", mask);

		int numSeeds = seeds.size();

		for (int i = 0; i < numSeeds; ++i)
		{
			//cv::floodFill(fill, seeds[i], cv::Scalar(255), 0, cv::Scalar(), cv::Scalar(), 8);
			cv::floodFill(fill, mask, seeds[i], cv::Scalar(255), 0, cv::Scalar(), cv::Scalar(), 8);
		}

		//cv::imwrite("fill.bmp", fill);

		cv::bitwise_and(weakEdgeMap, fill, dst);

		thin(dst, dst);//*/
	}
}