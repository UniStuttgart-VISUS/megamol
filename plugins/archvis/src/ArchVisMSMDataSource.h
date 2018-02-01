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

#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"

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

		virtual bool getDataCallback(megamol::core::Call& caller);
		
		/**
		* Loads the specified geometry and shader file
		*
		* @param shader_filename The shader file to load
		* @param geometry_filename The geometry file to load
		* @param nodeElement_filenamen The node-element table file
		*
		* @return True on success
		*/
		virtual bool load(std::string const& shader_filename,
			std::string const& geometry_filename,
			std::string const& nodesElement_filename);

	private:

		typedef std::tuple<float, float, float> Node;
		typedef std::tuple<int, int, int, int> FloorElement;
		typedef std::tuple<int, int> BeamElement;
		typedef std::tuple<int, int> DiagonalElement;

		typedef vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> Mat4x4;
		typedef vislib::math::Vector<float, 3> Vec3;
		typedef vislib::math::Quaternion<float> Quat;

		Mat4x4 computeElementTransform(Node src, Node tgt);

		void parseNodeElementTable(
			std::string const& filename,
			std::vector<Node>& nodes,
			std::vector<FloorElement>& floor_elements,
			std::vector<BeamElement>& beam_elements,
			std::vector<DiagonalElement>& diagonal_elements);

		std::vector<std::string> parsePartsList(std::string const& filename);


		/** The shader file name */
		megamol::core::param::ParamSlot m_shaderFilename_slot;

		/** The mesh list file name */
		megamol::core::param::ParamSlot m_partsList_slot;

		/** The node/element list file name */
		megamol::core::param::ParamSlot m_nodeElement_table_slot;

		/** The IP Adress for sensor data transfer */
		megamol::core::param::ParamSlot m_IPAdress_slot;

		/** The socket that receives the sensor data */
		vislib::net::Socket m_sensor_data_socket;
	};
}
}

#endif