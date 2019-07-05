/*
* DebugGPUMaterialDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef DEBUG_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED
#define DEBUG_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mesh/AbstractGPUMaterialDataSource.h"
#include "mesh/GPUMaterialCollection.h"

namespace megamol
{
	namespace mesh
	{
		class DebugGPUMaterialDataSource : public AbstractGPUMaterialDataSource
		{
		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "DebugGPUMaterialDataSource";
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


			DebugGPUMaterialDataSource();
			~DebugGPUMaterialDataSource();

		protected:

			virtual bool create();

			virtual bool getDataCallback(core::Call& caller);

			/**
			* Generate material data for debugging, i.e. load a fixed btf shader
			*
			* @return True on success
			*/
			virtual bool load();
		};
	}
}

#endif // !DEBUG_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED
