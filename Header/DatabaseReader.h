/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#include <string>
#include <fstream>

namespace haft
{
	class DatabaseReader
	{
	public:
		virtual ~DatabaseReader()
		{
		}

		virtual bool grabNext(cv::Mat& image, cv::Mat& ground) = 0;
		virtual unsigned int getSize() const = 0;
	};

	class JonesDatabaseReader : public DatabaseReader
	{
	private:
		std::ifstream file;
		std::string path;
		unsigned int size;

	public:
		JonesDatabaseReader(std::string path) : path(path)
		{
			file.open(path + "/skin-list.txt");
			size = 4665;
		}

		bool grabNext(cv::Mat& image, cv::Mat& ground) override
		{
			bool succeeded = true;

			if (!file.eof())
			{
				std::string imageID;
				file >> imageID;
				std::string inputURL = "";
				std::string groundURL = "";
				inputURL = inputURL + path + "/skin-images/" + imageID + ".jpg";
				groundURL = groundURL + path + "/masks/" + imageID + ".pbm";
				image = cv::imread(inputURL);
				cv::Mat temp = cv::imread(groundURL);

				//if ground type is not one uchar channel, then transform it because cv::calcHist need CV_8UC1 masks
				if (temp.type() != CV_8UC1)
				{
					ground = cv::Mat::zeros(temp.rows, temp.cols, CV_8UC1);
					int tempSize = temp.rows * temp.cols * temp.channels();
					int j = 0;
					for (int i = 0; i < tempSize; i += temp.channels())
					{
						if (temp.data[i])
							ground.data[j] = 255;

						++j;
					}
				}
				else
				{
					ground = temp.clone();
				}
			}
			else
			{
				succeeded = false;
			}

			return succeeded;
		}
		unsigned int getSize() const override
		{ return size; }
	};	
}