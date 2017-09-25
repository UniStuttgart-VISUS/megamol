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

#include "mmcore/param/ParamSlot.h"

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

		virtual bool getDataCallback(core::Call& caller);

		/**
		* Loads the specified geometry and shader file
		*
		* @param shader_filename The shader file to load
		* @param geometry_filename The geometry file to load
		*
		* @return True on success
		*/
		virtual bool load(std::string const& shader_filename, std::string const& geometry_filename);

	private:

		/** The mesh file name */
		core::param::ParamSlot m_geometryFilename_slot;

		/** The shader file name */
		core::param::ParamSlot m_shaderFilename_slot;
	};
}
}

#endif