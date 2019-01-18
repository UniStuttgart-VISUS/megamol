/*
* DebugMaterialsDataSource.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef DEBUG_MATERIALS_DATA_SOURCE_H_INCLUDED
#define DEBUG_MATERIALS_DATA_SOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh/AbstractMaterialsDataSource.h"
#include "ng_mesh/MaterialDataStorage.h"

namespace megamol
{
	namespace ngmesh
	{
		class DebugMaterialsDataSource : public AbstractMaterialsDataSource
		{
		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "DebugMaterialsDataSource";
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


			DebugMaterialsDataSource();
			~DebugMaterialsDataSource();

		protected:

			virtual bool getDataCallback(core::Call& caller);

			/**
			* Generat mesh data for debugging BatchedMeshesDataCall and rendering
			*
			* @return True on success
			*/
			virtual bool load();

		private:

			std::shared_ptr<MaterialsDataStorage> m_material_storage;
		};
	}
}

#endif // !DEBUG_MATERIALS_DATA_SOURCE_H_INCLUDED
