/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef NG_MESH_DEBUG_DATASOURCE_H_INCLUDED
#define NG_MESH_DEBUG_DATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ng_mesh/AbstractNGMeshDataSource.h"

namespace megamol {
namespace ngmesh {

	class NGMeshDebugDataSource : public AbstractNGMeshDataSource
	{
	public:
		/**
		* Answer the name of this module.
		*
		* @return The name of this module.
		*/
		static const char *ClassName(void) {
			return "NGMeshDebugDataSource";
		}

		/**
		* Answer a human readable description of this module.
		*
		* @return A human readable description of this module.
		*/
		static const char *Description(void) {
			return "Data source for debuging NGMeshRenderer";
		}

		/**
		* Answers whether this module is available on the current system.
		*
		* @return 'true' if the module is available, 'false' otherwise.
		*/
		static bool IsAvailable(void) {
			return true;
		}

		NGMeshDebugDataSource();
		~NGMeshDebugDataSource();

	protected:

		virtual bool load(std::string const& filename);

	private:

	};
}
}

#endif