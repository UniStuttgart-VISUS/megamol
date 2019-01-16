/*
* AbstractBatchedMeshesDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef ABSTRACT_BATCHED_MESHES_DATA_SOURCE_H_INCLUDED
#define ABSTRACT_BATCHED_MESHES_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <array>

#include "mmcore/CalleeSlot.h"
#include "ng_mesh.h"

namespace megamol
{
	namespace ngmesh
	{
		class NG_MESH_API AbstractBatchedMeshesDataSource : public core::Module
		{
		public:
			AbstractBatchedMeshesDataSource();
			virtual ~AbstractBatchedMeshesDataSource();

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
			 * The bounding box stored as left,bottom,back,right,top,front
			 */
			std::array<float,6> m_bbox;

		private:

			/** The slot for requesting data */
			megamol::core::CalleeSlot m_getData_slot;
		};
	}
}



#endif // !ABSTRACT_BATCHED_MESHES_DATA_SOURCE_H_INCLUDED
