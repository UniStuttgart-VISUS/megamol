/*
* glTFRenderTasksDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GLTF_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#define GLTF_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"

#include "mesh/AbstractGPURenderTaskDataSource.h"
#include "mesh/CallGltfData.h"

namespace megamol
{
	namespace mesh
	{
		class GlTFRenderTasksDataSource : public AbstractGPURenderTaskDataSource
		{
		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "GlTFRenderTasksDataSource";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Data source for loading render tasks based on the nodes of a glTF file";
			}

			/**
			* Answers whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}


			GlTFRenderTasksDataSource();
			~GlTFRenderTasksDataSource();

		protected:

			virtual bool getDataCallback(core::Call& caller);

		private:

			megamol::core::CallerSlot m_glTF_callerSlot;
		};
	}
}

#endif // !GLTF_RENDER_TASK_DATA_SOURCE_H_INCLUDED
