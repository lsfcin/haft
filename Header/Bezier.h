/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "Util.h"

namespace haft
{
	class Bezier
	{
	public:
		static void sampleCurvePoints(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, std::vector<cv::Point2f>& points, unsigned int sampling = 100)
		{
			double t = 0;
			const double increment = 1.0 / sampling;

			cv::Point2f b01, b02, b11;

			for (int i = 0; i < sampling; ++i)
			{
				t += increment;

				b01 = (1.0 - t) * p1 + t * p2;
				b02 = (1.0 - t) * p2 + t * p3;
				b11 = (1.0 - t) * b01 + t * b02;

				points.push_back(b11);
			}
		}
		static void sampleCurvePoints(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4, std::vector<cv::Point2f>& points, unsigned sampling = 100)
		{
			double t = 0;
			const double increment = 1.0 / sampling;

			cv::Point2f b01, b02, b03, b11, b12, b21;

			for (int i = 0; i < sampling; ++i)
			{
				t += increment;

				b01 = (1.0 - t) * p1 + t * p2;
				b02 = (1.0 - t) * p2 + t * p3;
				b03 = (1.0 - t) * p3 + t * p4;
				b11 = (1.0 - t) * b01 + t * b02;
				b12 = (1.0 - t) * b02 + t * b03;
				b21 = (1.0 - t) * b11 + t * b12;

				points.push_back(b21);
			}
		}
		static void sampleSplinesPoints(const std::vector<cv::Point>& inputControlPoints, std::vector<cv::Point2f>& points, bool closed = false)
		{
			cv::Point2f p1, p2, p3, p4;
			unsigned int sampling;
			const unsigned int size = inputControlPoints.size();

			unsigned int i;

			//first spline sampling
			if (!closed)
			{
				p1 = inputControlPoints[0];
				p2 = inputControlPoints[1];
				p3 = inputControlPoints[2];
				sampling = Util::distance4C(p1, p2) + Util::distance4C(p2, p3);
				sampleCurvePoints(p1, p2, p3, points, sampling);
				i = 2;
			}
			else
			{
				i = 0;
			}

			//mid splines iteration
			for (; i < size - 5; i += 3)
			{
				p1 = inputControlPoints[i];
				p2 = inputControlPoints[i + 1];
				p3 = inputControlPoints[i + 2];
				p4 = inputControlPoints[i + 3];
				sampling = Util::distance4C(p1, p2) + Util::distance4C(p2, p3) + Util::distance4C(p3, p4);
				sampleCurvePoints(p1, p2, p3, p4, points, sampling);
			}

			//last spline sampling
			if (!closed)
			{
				p1 = inputControlPoints[size - 3];
				p2 = inputControlPoints[size - 2];
				p3 = inputControlPoints[size - 1];
				sampling = Util::distance4C(p1, p2) + Util::distance4C(p2, p3);
				sampleCurvePoints(p1, p2, p3, points, sampling);
			}
			else
			{
				p1 = inputControlPoints[size - 3];
				p2 = inputControlPoints[size - 2];
				p3 = inputControlPoints[size - 1];
				p4 = inputControlPoints[0];
				sampling = Util::distance4C(p1, p2) + Util::distance4C(p2, p3) + Util::distance4C(p3, p4);
				sampleCurvePoints(p1, p2, p3, p4, points, sampling);
			}
		}
		static void calcNewControlPoints(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, cv::Point2f& b1, cv::Point2f& b2)
		{
			//calcula o vetor v entre os pontos p1 e p3
			cv::Point2f v = p1 - p3;

			//calcula as distâncias d1 entre p1 e p2, e d2 entre p2 e p3
			float d1 = Util::distanceEuclidean(p1, p2);
			float d2 = Util::distanceEuclidean(p2, p3);

			//calcula os dois pares possíveis de b1 e b2:

			//cria-se v1 = v com norma d1/3
			cv::Point2f v1 = v;
			Util::setNorm(v1, d1 / 3);

			//cria-se v2 = v com norma d2/3
			cv::Point2f v2 = v;
			Util::setNorm(v2, d2 / 3);

			//calcula b1' como p2 + v1
			//calcula b2' como p2 + v2
			//calcula b1" como p2 - v1
			//calcula b2" como p2 - v2
			cv::Point b1¹ = p2 + v1;
			cv::Point b2¹ = p2 - v2;
			cv::Point b1² = p2 + v1;
			cv::Point b2² = p2 - v2;

			//calcula a soma das distâncias sum1 entre b1' e p1 e b2' e p3
			//calcula a soma das distâncias sum2 entre b1" e p1 e b2" e p3
			float sum1 = Util::distance4C(b1¹, p1) + Util::distance4C(b2¹, p3);
			float sum2 = Util::distance4C(b1², p1) + Util::distance4C(b2², p3);

			//se sum1 for menor que sum2 indica que os pontos b1' e b2' estão corretos
			if (sum1 < sum2)
			{
				b1 = b1¹;
				b2 = b2¹;
			}
			//se não, b1" e b2" é que estão corretos
			else
			{
				b1 = b1²;
				b2 = b2²;
			}
		}
		static void calcBezierSplinesControlPoints(const std::vector<cv::Point>& inputControlPoints, std::vector<cv::Point>& outputControlPoints, bool closed = false)
		{
			outputControlPoints.clear();

			//pega 3 pontos
			cv::Point2f p1, p2, p3, b1, b2;

			//if not closed, the first point will not be added by the for statement below, so do it here
			if (!closed) outputControlPoints.push_back(inputControlPoints[0]);

			//add triads of points b1, p2, b2
			const unsigned int size = inputControlPoints.size();
			for (unsigned int i = 0; i < size - 2; ++i)
			{
				p1 = inputControlPoints[i];
				p2 = inputControlPoints[i + 1];
				p3 = inputControlPoints[i + 2];

				calcNewControlPoints(p1, p2, p3, b1, b2);
				outputControlPoints.push_back(b1);
				outputControlPoints.push_back(p2);
				outputControlPoints.push_back(b2);
			}

			//in case the points complete a closed contour, create splines with the end and the start points
			if (closed)
			{
				p1 = inputControlPoints[size - 2];
				p2 = inputControlPoints[size - 1];
				p3 = inputControlPoints[0];
				calcNewControlPoints(p1, p2, p3, b1, b2);
				outputControlPoints.push_back(b1);
				outputControlPoints.push_back(p2);
				outputControlPoints.push_back(b2);

				p1 = inputControlPoints[size - 1];
				p2 = inputControlPoints[0];
				p3 = inputControlPoints[1];
				calcNewControlPoints(p1, p2, p3, b1, b2);
				outputControlPoints.push_back(b1);
				outputControlPoints.push_back(p2);
				outputControlPoints.push_back(b2);
			}

			//if not closed, add the end point to output
			if (!closed) outputControlPoints.push_back(inputControlPoints[size - 1]);
		}
	};
}