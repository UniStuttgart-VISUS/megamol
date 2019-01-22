/*
* DebugBatchedMeshesDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef DEBUG_BATCHED_MESHES_DATA_SOURCE_H_INCLUDED
#define DEBUG_BATCHED_MESHES_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh/AbstractBatchedMeshesDataSource.h"
#include "ng_mesh/MeshDataStorage.h"

namespace megamol
{
	namespace ngmesh
	{
		class DebugBatchedMeshesDataSource : public AbstractBatchedMeshesDataSource
		{
		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "DebugBatchedMeshesDataSource";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Data source for debuging NGMeshRenderer & NGMesh data calls";
			}

			/**
			* Answers whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}


			DebugBatchedMeshesDataSource();
			~DebugBatchedMeshesDataSource();

		protected:

			virtual bool getDataCallback(core::Call& caller);

			/**
			* Generat mesh data for debugging BatchedMeshesDataCall and rendering
			*
			* @return True on success
			*/
			virtual bool load();

		private:

			MeshDataStorage           m_mesh_data_storage;
			BatchedMeshesDataAccessor m_mesh_data_accessor;

			bool                      m_data_loaded;
		};
	}
}

#endif // !DEBUG_BATCHED_MESHES_DATA_SOURCE_H_INCLUDED
