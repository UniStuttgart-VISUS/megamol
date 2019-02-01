/*
* AbstractGPUMaterialDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef ABSTRACT_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "ng_mesh/ng_mesh.h"
#include "GPUMaterialDataStorage.h"

namespace megamol
{
	namespace ngmesh
	{
		class NG_MESH_API AbstractGPUMaterialDataSource : public core::Module
		{
		public:
			AbstractGPUMaterialDataSource();
			virtual ~AbstractGPUMaterialDataSource();

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
			* Implementation of 'Release'.
			*/
			virtual void release();

		private:
			GPUMaterialDataStorage m_gpu_materials;

			/** The slot for requesting data */
			megamol::core::CalleeSlot m_getData_slot;
		};
	}
}

#endif // !ABSTRACT_GPU_MATERIAL_DATA_SOURCE_H_INCLUDED