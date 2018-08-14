/*
 * STLWriter.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_IO_STLWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_STLWRITER_H_INCLUDED
#pragma once

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/DataWriterCtrlCall.h"

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/CallTriMeshData.h"

#ifdef MEGAMOL_NG_MESH
#include "ng_mesh/CallNGMeshRenderBatches.h"
#endif

#include <string>
#include <tuple>

namespace megamol
{
	namespace stdplugin
	{
		namespace datatools
		{
			namespace io
			{
				class STLWriter : public core::AbstractDataWriter
				{
				public:
					/// <summary>
					/// Answer the name of this module
					/// </summary>
					/// <returns>Module name</returns>
					static const char* ClassName()
					{
						return "STLWriter";
					}

					/// <summary>
					/// Answer a human-readable description
					/// </summary>
					/// <returns>Description</returns>
					static const char* Description()
					{
						return "Writer for triangle data in STL files";
					}

					/// <summary>
					/// Answer the availability of this module
					/// </summary>
					/// <returns>Availability</returns>
					static bool IsAvailable()
					{
						return true;
					}

					/// <summary>
					/// Disallow quickstart (I don't know why, though)
					/// </summary>
					/// <returns>Quickstart supported?</returns>
					static bool SupportQuickstart()
					{
						return false;
					}

					/// <summary>
					/// Constructor
					/// </summary>
					STLWriter();

					/// <summary>
					/// Destructor
					/// </summary>
					~STLWriter();

				protected:

					/// <summary>
					/// Create the module
					/// </summary>
					/// <returns>True on success; false otherwise</returns>
					virtual bool create() override;

					/// <summary>
					/// Callback function for requesting capability information
					/// </summary>
					/// <param name="caller">Call for this request</param>
					/// <returns>True on success; false otherwise</returns>
					virtual bool getCapabilities(core::DataWriterCtrlCall& call) override;

					/// <summary>
					/// Callback function for storing data to file
					/// </summary>
					/// <returns>True on success; false otherwise</returns>
					virtual bool run() override;

					/// <summary>
					/// Release the module
					/// </summary>
					virtual void release() override;

				private:
					/// <summary>
					/// Get number of triangles, and float pointer to vertices and normals from mesh call
					/// </summary>
					/// <param name="call">Call where the data is stored in</param>
					/// <returns>[Number of triangles, vertex pointer, normal pointer]</returns>
					std::tuple<std::size_t, const float*, const float*> get_mesh_data(geocalls::CallTriMeshData& call) const;

#ifdef MEGAMOL_NG_MESH
					/// <summary>
					/// Get number of triangles, and float pointer to vertices and normals from NG mesh call
					/// </summary>
					/// <param name="call">Call where the data is stored in</param>
					/// <returns>[Number of triangles, vertex pointer, normal pointer]</returns>
					std::tuple<std::size_t, const float*, const float*> get_ngmesh_data(ngmesh::CallNGMeshRenderBatches& call) const;
#endif

					/// <summary>
					/// Write a binary file
					/// </summary>
					/// <param name="filename">File name of the STL file</param>
					/// <param name="mesh">Mesh data and information of the form [number of triangles, vertex pointer, normal pointer]</param>
					void write_binary(const std::string& filename, const std::tuple<std::size_t, const float*, const float*>& mesh) const;

					/// <summary>
					/// Write a textual file
					/// </summary>
					/// <param name="filename">File name of the STL file</param>
					/// <param name="mesh">Mesh data and information of the form [number of triangles, vertex pointer, normal pointer]</param>
					void write_ascii(const std::string& filename, const std::tuple<std::size_t, const float*, const float*>& mesh) const;

					/// File name
					core::param::ParamSlot filename_slot;

					/// Option for ASCII/binary
					core::param::ParamSlot ascii_binary_slot;

					/// Input
					core::CallerSlot mesh_input_slot;

#ifdef MEGAMOL_NG_MESH
					core::CallerSlot ngmesh_input_slot;
#endif
				};
			}
		}
	}
}

#endif // !MEGAMOL_DATATOOLS_IO_STLWRITER_H_INCLUDED