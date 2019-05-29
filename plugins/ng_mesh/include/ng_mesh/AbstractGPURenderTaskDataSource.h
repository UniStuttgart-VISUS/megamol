/*
* AbstractGPURenderTaskDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef ABSTRACT_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_GPU_RENDER_TASK_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "ng_mesh.h"
#include "GPURenderTaskCollection.h"

namespace megamol
{
	namespace ngmesh
	{
		class NG_MESH_API AbstractGPURenderTaskDataSource : public core::Module
		{
		public:
			AbstractGPURenderTaskDataSource();
			virtual ~AbstractGPURenderTaskDataSource();

		protected:
			/**
			* Implementation of 'Create'.
			*
			* @return 'true' on success, 'false' otherwise.
			*/
			virtual bool create(void);

			/**
			* Gets the data from the source.
			*
			* @param caller The calling call.
			*
			* @return 'true' on success, 'false' on failure.
			*/
			virtual bool getDataCallback(core::Call& caller) = 0;

			/**
			* Gets the data from the source.
			*
			* @param caller The calling call.
			*
			* @return 'true' on success, 'false' on failure.
			*/
			virtual bool getExtentCallback(core::Call& caller);

			/**
			* Implementation of 'Release'.
			*/
			virtual void release();

			/**
			* 
			*/
			std::shared_ptr<GPURenderTaskCollection> m_gpu_render_tasks;

			/** The slot for requesting data from this module, i.e. lhs connection */
			megamol::core::CalleeSlot m_getData_slot;

            /** The slot for querying mesh data, i.e. a rhs connection */
			megamol::core::CallerSlot m_mesh_callerSlot;

            /** The slot for querying material data, i.e. a rhs connection */
			megamol::core::CallerSlot m_material_callerSlot;
		};
	}
}

#endif // !ABSTRACT_RENDER_TASK_DATA_SOURCE_H_INCLUDED
