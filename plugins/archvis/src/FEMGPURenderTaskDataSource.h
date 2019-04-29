/*
* FEMGPURenderTaskDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef FEM_GPU_TASK_DATA_SOURCE_H_INCLUDED
#define FEM_GPU_TASK_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh/AbstractGPURenderTaskDataSource.h"

namespace megamol
{
	namespace archvis
	{
		class FEMGPURenderTaskDataSource : public ngmesh::AbstractGPURenderTaskDataSource
		{
		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "FEMGPURenderTasksDataSource";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Data source for loading render tasks based on FEM data";
			}

			/**
			* Answers whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}


			FEMGPURenderTaskDataSource();
			~FEMGPURenderTaskDataSource();

		protected:

			virtual bool getDataCallback(core::Call& caller);
		};
	}
}

#endif // !FEM_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
