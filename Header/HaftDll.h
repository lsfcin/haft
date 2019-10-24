/*
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.txt', which is part of this source code package.
 */

#pragma once

#ifdef HAFTDLL_EXPORTS
#define HAFTDLL_API __declspec(dllexport) 
#else
#define HAFTDLL_API __declspec(dllimport) 
#endif