/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <opencv2\opencv.hpp>

namespace haft
{
#define CV_BGR -1
	
	class PixelClassifier
	{
	public:
		virtual ~PixelClassifier()
		{
		}

		// returns a value between 0.0f and 1.0f as the classification result
		virtual inline float classify(uchar c1, uchar c2, uchar c3) const = 0;
		
		// returns a conversion code of the classifier. eg.: CV_BGR2HSV for HSV 
		// needed images. CV_BGR if no conversion needed. 
		virtual inline int getBGRConversion() const = 0;
		
		// overridable function available to implement classifier adaptability
		virtual inline void update(
			const cv::Mat& input,
			const cv::Mat& positiveMask,
			const cv::Mat& negativeMask,
			const float learningRate,
			const float threshold = 0.1) {}

		virtual inline bool readyToSegment() { return false; };
	};

	//SKIN SINGLE PIXEL VALIDATION FUNCTIONS
	inline bool RGBGoogleFixed(float b, float g, float r) //google's patent US 8.055.067 B2 RGB
	{
		float saturation = sqrt(pow(r, 2)*0.27847 - r*g*0.30610 +
			pow(g, 2)*0.28503 - r*b*0.25005 +
			pow(b, 2)*0.25661 - g*b*0.26317);
		float k;
		//if(lowThreshold) k = 0.4*(saturation+5); //low false-negatives
		//else             k = 1.0*(saturation+6.5); //low false-positives
		k = 0.53*(saturation + 5); //original

		return (r > g + k && r > b + k && g > 0 && r / b < 2.75 && r / g < 1.5);
	}
	inline bool RGBKovacetalFixed(float b, float g, float r) //kovac et al.
	{
		//if(lowThreshold) return (r > 30 && g > 15 && b > 15 && r + 10 > b && r > g && r/b < 3.5); //low false-negatives
		//else             return (r > 30 && g > 15 && b > 15 && r > b + 20 && r > g + 15 && r/b < 2.75 && r/g < 1.5); //low false-positives
		return (r > 95 && g > 40 && b > 20 && r > b && r > g &&
			(std::max(r, std::max(g, b)) - std::min(r, std::min(g, b)) > 15) &&
			r - g > 15) ||
			(r > 220 && g > 210 && b > 170 && abs(r - g) <= 15); //original
	}
	inline bool RGBGomezMoralesFixed(float b, float g, float r) //gomez and morales
	{
		float rn, gn, bn; //normalized
		const bool notRed = r / b < 3.5 && r / g < 1.5;
		const bool notGrey = r - g > 15;
		rn = r / (r + g + b); gn = g / (r + g + b); bn = b / (r + g + b); r = rn; g = gn; b = bn;

		//if(lowThreshold) return (r/g > 1.0 && (r*b)/pow(r+g+b,2) > 0.065 && (r*g)/pow(r+g+b,2) > 0.105); //low false-negatives
		//else             return (r/g > 1.185 && (r*b)/pow(r+g+b,2) > 0.07 && (r*g)/pow(r+g+b,2) > 0.112) && notRed && notGrey; //low false-positives
		return (r / g > 1.185 && (r*b) / pow(r + g + b, 2) > 0.107 && (r*g) / pow(r + g + b, 2) > 0.112); //original
	}
	inline bool HSVTsekeridouPitasFixed(float h, float s, float v) //tsekeridou and pitas
	{
		float hn, sn, vn; //converted
		hn = h * 2;
		sn = s / 255;
		vn = v;

		//if(lowThreshold) return (vn > 40 && sn > 0.02 && sn < 0.95 && (hn < 80 || hn > 335)); //low false-negatives
		//else             return (vn > 40 && vn < 250 && sn > 0.2 && sn < 0.7 && (hn < 30 || hn > 335)); //low false-positives
		return (vn > 102 && sn >= 0.2 && sn <= 0.6 && (hn <= 25 || hn >= 335)); //original
	}
	inline bool HSVSobottkaPitasFixed(float h, float s, float v) //K. Sobottka and I. Pitas
	{
		float hn, sn, vn; //converted
		hn = h * 2;
		sn = s / 255;
		return 0.23 <= sn && sn <= 0.68 && 0 <= hn && hn <= 50;
	}
	inline bool HSVOpenCVASDFixed(float h, float s, float v) //opencv adaptive skin segmenter
	{
		//if(lowThreshold) return v >= 10 && v <= 250 && h >= 1 && h <= 33; //low false-negatives
		//else             return v >= 40 && v <= 250 && h >= 3 && h <= 15 && s > 50 && s < 180; //low false-positives
		return v >= 15 && v <= 250 && h >= 3 && h <= 33;  //original
	}
	inline bool YCrCbChaiNgunFixed(float y, float cr, float cb) //chai and ngun
	{
		return 77 <= cb && cb < 127 && 133 <= cr && cr < 173;
	}
	inline bool YCrCbGarciaTziritasFixed(float y, float cr, float cb)
	{
		double theta1;
		double theta2;
		double theta3;
		double theta4;

		if (y > 128)
		{
			theta1 = -2 + ((256 - y) / 16);
			theta2 = 20 - ((356 - y) / 16);
			theta3 = 6;
			theta4 = -8;
		}
		else
		{
			theta1 = 6;
			theta2 = 12;
			theta3 = 2 + (y / 32);
			theta4 = -16 + (y / 16);
		}

		cr -= 128;
		cb -= 128;

		bool c1 = cr >= -2 * (cb + 24); bool c2 = cr >= -(cb + 17);
		bool c3 = cr >= -4 * (cb + 32); bool c4 = cr >= 2.5  * (cb + theta1);
		bool c5 = cr >= theta3;        bool c6 = cr >= 0.5  * (theta4 - cb);
		bool c7 = cr <= (220 - cb) / 6; bool c8 = cr <= (4 / 3) * (theta2 - cb);

		return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8;
	}

	//PURE COLORS SINGLE PIXEL VALIDATION FUNCTIONS
	inline bool RGBBlack(float b, float g, float r)
	{
		const int t = 10;
		return r < t && g < t && b < t;
	}
	inline bool RGBRed(float b, float g, float r)
	{
		return r > g * 2 && r > b * 3;
	}
	inline bool HSVRed(float h, float s, float v)
	{
		return ((h >= 0 && h < 3) || (h > 167)) && s > 170;
	}
	inline bool HSVYellow(float h, float s, float v)
	{
		return h > 15 && h < 30 && s > 130;
	}
	inline bool HSVPink(float h, float s, float v)
	{
		return (h > 150 && h < 177 && s > 200) || (h > 150 && h < 177 && s > 80 && v > 180);
	}
	inline bool HSVGreen(float h, float s, float v)
	{
		return h > 38 && h < 64 && s > 85 && s < 160;
	}
	inline bool HSVBlue(float h, float s, float v)
	{
		return h > 95 && h < 115 && s > 180 && v > 50;
	}
	inline bool HSVPurple(float h, float s, float v)
	{
		return h > 115 && h < 150 && v > 60 && s > 100;// && s < 100 && v > 50;
	}	
}