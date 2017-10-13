/*
* ArchVisMSMDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef ARCH_VIS_MSM_DATASOURCE_H_INCLUDED
#define ARCH_VIS_MSM_DATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/net/Socket.h"

#include "mmcore/param/ParamSlot.h"

#include "ng_mesh/AbstractNGMeshDataSource.h"

using namespace megamol::ngmesh;

namespace megamol {
namespace archvis {

	class ArchVisMSMDataSource : public AbstractNGMeshDataSource
	{
	public:
		/**
		* Answer the name of this module.
		*
		* @return The name of this module.
		*/
		static const char *ClassName(void) {
			return "ArchVisMSMDataSource";
		}

		/**
		* Answer a human readable description of this module.
		*
		* @return A human readable description of this module.
		*/
		static const char *Description(void) {
			return "Data source for visualizing SFB1244's 'Maﬂstabsmodell'";
		}

		/**
		* Answers whether this module is available on the current system.
		*
		* @return 'true' if the module is available, 'false' otherwise.
		*/
		static bool IsAvailable(void) {
			return true;
		}

		ArchVisMSMDataSource();
		~ArchVisMSMDataSource();

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

		/** The shader file name */
		core::param::ParamSlot m_shaderFilename_slot;

		/** The mesh file name */
		core::param::ParamSlot m_geometryFilename_slot;

		/** The IP Adress for sensor data transfer */
		core::param::ParamSlot m_IPAdress_slot;

		/** The socket that receives the sensor data */
		vislib::net::Socket m_sensor_data_socket;
	};
}
}

#endif