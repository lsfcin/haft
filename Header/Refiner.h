/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "PRFLabeler.h"
#include "UI.h"
#include "Target.h"

namespace haft
{
	class Refiner
	{
	public:
		static bool refine(const cv::Mat& mask, const cv::Point2f& startPoint, Target& target)
		{
			auto succeeded = false;

			const cv::Size imageSize = mask.size();

			cv::Mat pointsImage = cv::Mat::zeros(imageSize, CV_8UC1);
			auto count = 0;
			target.midPoint = cv::Point2d(0, 0);

			for (int j = target.roi.y; j < target.roi.y + target.roi.height; ++j)
			{
				for (int i = target.roi.x; i < target.roi.x + target.roi.width; ++i)
				{
					int pointID = Util::index1D(cv::Point2d(i, j), mask.size());
					int maskData = mask.data[pointID];
					if (maskData == target.contourID + CONTOUR_VISUALIZATION_OFFSET)
					{
						++count;
						target.midPoint.x += i;
						target.midPoint.y += j;
					}
				}
			}

			if (count > 0)
			{
				target.midPoint.x /= count;
				target.midPoint.y /= count;
				target.rotatedBOX = findRotatedROI(imageSize, cv::Mat(pointsImage, target.roi), cv::Point2f(target.roi.x, target.roi.y));
				succeeded = true;
			}

			return succeeded;
		}

		static cv::RotatedRect findRotatedROI(const cv::Size& bigImageSize, const cv::Mat& subImage, const cv::Point2f& offset)
		{
			const unsigned int TOLERANCE = 10;
			double m00 = 0, m10, m01, mu20, mu11, mu02, inv_m00;
			double a, b, c, xc, yc;
			double rotate_a, rotate_c;
			double theta = 0, square;
			double cs, sn;
			double length = 0, width = 0;

			cv::Moments moments = cv::moments(subImage);

			m00 = moments.m00;
			m10 = moments.m10;
			m01 = moments.m01;
			mu11 = moments.mu11;
			mu20 = moments.mu20;
			mu02 = moments.mu02;

			if (fabs(m00) < DBL_EPSILON)
				return cv::RotatedRect();

			inv_m00 = 1. / m00;
			xc = cvRound(m10 * inv_m00 + offset.x);
			yc = cvRound(m01 * inv_m00 + offset.y);
			a = mu20 * inv_m00;
			b = mu11 * inv_m00;
			c = mu02 * inv_m00;

			/* Calculating width & height */
			square = sqrt(4 * b * b + (a - c) * (a - c));

			/* Calculating orientation */
			theta = atan2(2 * b, a - c + square);

			/* Calculating width & length of figure */
			cs = cos(theta);
			sn = sin(theta);

			rotate_a = cs * cs * mu20 + 2 * cs * sn * mu11 + sn * sn * mu02;
			rotate_c = sn * sn * mu20 - 2 * cs * sn * mu11 + cs * cs * mu02;
			length = sqrt(rotate_a * inv_m00) * 4;
			width = sqrt(rotate_c * inv_m00) * 4;

			/* In case, when tetta is 0 or 1.57... the Length & Width may be exchanged */
			if (length < width)
			{
				double t;
				CV_SWAP(length, width, t);
				CV_SWAP(cs, sn, t);
				theta = CV_PI*0.5 - theta;
			}

			unsigned int t0, t1;
			unsigned int _xc = cvRound(xc);
			unsigned int _yc = cvRound(yc);

			t0 = cvRound(fabs(length * cs));
			t1 = cvRound(fabs(width * sn));

			t0 = MAX(t0, t1) + 2;

			cv::Rect rect;
			rect.width = MIN(t0, (bigImageSize.width - _xc) * 2);

			t0 = cvRound(fabs(length * sn));
			t1 = cvRound(fabs(width * cs));

			t0 = MAX(t0, t1) + 2;
			rect.height = MIN(t0, (bigImageSize.height - _yc) * 2);

			rect.x = MAX(0, _xc - rect.width / 2);
			rect.y = MAX(0, _yc - rect.height / 2);

			rect.width = MIN(bigImageSize.width - rect.x, rect.width);
			rect.height = MIN(bigImageSize.height - rect.y, rect.height);

			float angle = float((CV_PI*0.5 + theta)*180. / CV_PI);
			while (angle < 0)    angle += 360;
			while (angle >= 360) angle -= 360;
			if (angle >= 180)    angle -= 180;

			cv::Point2f center = cvPoint2D32f(rect.x + rect.width*0.5f, rect.y + rect.height*0.5f);

			return cv::RotatedRect(cv::Point2f(center), cv::Size2f(length, width), angle + 90);
		}
	};
}

//cv::Mat rightHandMaskImage(currMask, target.roi);
//UI::showRectangle(roi);

/*
unsigned int count = 0;
while(Util::nonBlackCount(rightHandMaskImage) > 10)
{
++count;
erode(rightHandMaskImage, rightHandMaskImage, cv::Mat(), cv::Point(-1, -1));
cv::imwrite("test2.png", rightHandMaskImage);
}

erode(currMask, currMask, cv::Mat(), cv::Point(-1, -1), count);
UI::showImage(currMask);*/