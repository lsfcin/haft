/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

namespace haft
{
	class TargetIDManager
	{
	public:
		static TargetIDManager& getInstance()
		{
			static TargetIDManager instance; // Guaranteed to be destroyed. Instantiated on first use.
			return instance;
		}

		unsigned int getNextAvailableID()
		{
			++currentID;
			return currentID;
		}
	private:
		unsigned int currentID;

		TargetIDManager() // Constructor? (the {} brackets) are needed here.
		{
			currentID = -1;
		}
		// Dont forget to declare these two. You want to make sure they
		// are unaccessable otherwise you may accidently get copies of
		// your singleton appearing.
		TargetIDManager(TargetIDManager const&);        // Don't Implement
		void operator=(TargetIDManager const&); // Don't implement
	};
}