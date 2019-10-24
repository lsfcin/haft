/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <vector>

#include "Util.h"
#include "TargetIDManager.h"
#include "Features.h"

namespace haft
{
#define MAX_STORED 200

	enum TargetType
	{
		HAND,
		FACE,
		BODY,
		UNKNOWN
	};

	class Target
	{
	public:

		Target()
		{
			id = TargetIDManager::getInstance().getNextAvailableID();
			type = UNKNOWN;
		}

		/*Target(TargetType type, cv::Rect roi) :
			roi(roi), 
			succeeded(false), 
			type(type),
			handProbability(0), 
			faceProbability(-1),
			handDecrement(0), 
			faceDecrement(0)
		{
			midPoint = Util::midPoint(roi);
			calcMaxNumFeatures((roi.width << 1) + (roi.height << 1));
		}*/

		void calcMaxNumFeatures(int perimeter)
		{
			maxNumFeatures = perimeter >> 4;
		}

		void cropFeatures()
		{
			for (unsigned int i = 0; i < features.size(); ++i)
			{
				if (!Util::isInside(roi, features[i].p2D))
					features.remove(i);
			}
		}

		void update(Target const& target)
		{
			Target temp;
			temp.id = this->id;
			temp.contour = this->contour;
			temp.contourID = this->contourID;
			temp.features = this->features;
			temp.maxNumFeatures = this->maxNumFeatures;
			temp.medianPoint = this->medianPoint;
			temp.midPoint = this->midPoint;
			temp.roi = this->roi;
			temp.rotatedBOX = this->rotatedBOX;
			temp.succeeded = this->succeeded;
			temp.type = this->type;

			previous.push_back(temp);

			this->contour = target.contour;
			this->contourID = target.contourID;
			this->features = target.features;
			this->maxNumFeatures = target.maxNumFeatures;
			this->medianPoint = target.medianPoint;
			this->midPoint = target.midPoint;
			this->roi = target.roi;
			this->rotatedBOX = target.rotatedBOX;
			this->succeeded = target.succeeded;
			this->type = target.type;

			if (previous.size() > MAX_STORED)
				previous.erase(previous.begin());

			updateMaxNumFeatures();
		}

		void updateMaxNumFeatures()
		{
			std::vector<float> numFeaturesVec;
			for (auto i = 0; i < previous.size(); ++i)
			{
				numFeaturesVec.push_back(previous[i].contour.size());
			}
			numFeaturesVec.push_back(this->contour.size());

			auto smoothed = Util::smooth(numFeaturesVec);

			calcMaxNumFeatures(smoothed);
		}

		Features& lastFeatures()
		{
			if (previous.size() > 0)
				return previous.back().features;
			else
				return Features();
		}

		cv::Rect& lastROI()
		{
			if (previous.size() > 0)
				return previous.back().roi;
			else
				return cv::Rect(0, 0, 0, 0);
		}

		cv::Point2f lastMidPoint()
		{
			if (previous.size() > 0)
				return previous.back().midPoint;
			else
				return cv::Point2f(0, 0);
		}

		void setROI(cv::Rect inputROI)
		{
			roi = inputROI;
			//midPoint = Util::midPoint(roi);
			calcMaxNumFeatures((roi.width << 1) + (roi.height << 1));
		}

		unsigned int getID() const
		{
			return id;
		}


		TargetType getType() const
		{
			return type;
		}

		void setType(TargetType newType)
		{
			this->type = newType;
		}

		void increaseHandProbability(float increment)
		{
			handProbability += increment;
			if (handProbability >= 1)
			{
				type = HAND;
				handProbability = 1;
				faceProbability = 0;
			}
		}

		void increaseFaceProbability(float increment)
		{
			faceProbability += increment;
			if (faceProbability >= 1)
			{
				type = FACE;
				faceProbability = 1;
				handProbability = 0;
			}
		}

		void decreaseHandProbability(float increment)
		{
			handDecrement += increment;
			handProbability -= handDecrement;
		}

		void decreaseFaceProbability(float increment)
		{
			faceDecrement += increment;
			faceProbability -= faceDecrement;
		}

		float getHandProbability()
		{
			return handProbability;
		}

		float getFaceProbability()
		{
			return faceProbability;
		}

		std::vector<Target> previous;
		int maxNumFeatures;
		Features features;

		std::vector<cv::Point> contour;
		int contourID;

		cv::RotatedRect rotatedBOX;
		cv::Rect roi;

		cv::Point2f medianPoint = cv::Point2f(0.f, 0.f);
		cv::Point2f midPoint = cv::Point2f(0.f, 0.f);

		cv::Point2f position = cv::Point2f(0.f, 0.f);
		float scale = 1.f;

		bool succeeded;
	private:
		TargetType type;
		unsigned int id;
		float handProbability;
		float faceProbability;
		float handDecrement;
		float faceDecrement;
	};
}
