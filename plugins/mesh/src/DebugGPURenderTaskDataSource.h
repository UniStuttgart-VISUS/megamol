/*
* DebugGPURenderTaskDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef DEBUG_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#define DEBUG_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mesh/AbstractGPURenderTaskDataSource.h"

namespace megamol
{
	namespace mesh
	{
		class DebugGPURenderTaskDataSource : public AbstractGPURenderTaskDataSource
		{
		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "DebugGPURenderTaskDataSource";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Data source for debuging RenderMDIMesh & mesh GPU data calls";
			}

			/**
			* Answers whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}


			DebugGPURenderTaskDataSource();
			~DebugGPURenderTaskDataSource();

		protected:

			virtual bool getDataCallback(core::Call& caller);

			/**
			* Generat mesh data for debugging BatchedMeshesDataCall and rendering
			*
			* @return True on success
			*/
			virtual bool load();
		};
	}
}

#endif // !DEBUG_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
