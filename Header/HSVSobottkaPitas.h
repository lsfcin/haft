/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include "PixelClassifier.h"

namespace haft
{
	class HSVSobottkaPitas : public PixelClassifier
	{
	private:
		float lowS;
		float highS;
		float lowH;
		float highH;

	public:
		HSVSobottkaPitas() : lowS(0.23), highS(0.68), lowH(0), highH(50) {}
		inline void reset()
		{
			lowS = 0.23;
			highS = 0.68;
			lowH = 0;
			highH = 50;
		}
		inline void update(float closure = 0.5)
		{
			float alS, blS, ahS, bhS, alH, blH, ahH, bhH;

			if (closure <= 0.5)
			{
				alS = 0.46; blS = 0;
				ahS = -0.64; bhS = 1;
				alH = 310;  blH = -155;
				ahH = -310;  bhH = 205;
			}
			else
			{
				alS = 0.44; blS = 0.01;
				ahS = -0.44; bhS = 0.9;
				alH = 48;   blH = -24;
				ahH = -50;   bhH = 75;
			}

			lowS = alS * closure + blS;
			highS = ahS * closure + bhS;
			lowH = alH * closure + blH;
			highH = ahH * closure + bhH;
		}
		inline float classify(uchar h, uchar s, uchar v) const
		{
			float hn, sn, vn; //converted
			hn = h * 2;
			sn = s / 255;

			float result = 0.0f;

			if (lowH >= 0)
				if (lowS <= sn && sn <= highS && lowH <= hn && hn <= highH) result = 1.0f;
				else //in case low S threshold is below zero
					if (lowS <= sn && sn <= highS && (lowH + 360) <= hn || hn <= highH) result = 1.0f;

			return result;
		}
		inline int getBGRConversion() const { return CV_BGR2HSV_FULL; }
		inline bool readyToSegment() const { return true; }
	};
}