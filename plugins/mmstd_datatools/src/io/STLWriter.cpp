/*
 * STLWriter.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "STLWriter.h"

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/DataWriterCtrlCall.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/EnumParam.h"

#include "geometry_calls/CallTriMeshData.h"

#ifdef MEGAMOL_NG_MESH
#include "ng_mesh/CallNGMeshRenderBatches.h"
#endif

#include <fstream>
#include <sstream>
#include <stdexcept>
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
				STLWriter::STLWriter() : core::AbstractDataWriter()
					, filename_slot("STL file", "The name of to the STL file to write")
					, ascii_binary_slot("Output type", "Write an ASCII or binary file?")
					, mesh_input_slot("mesh_data", "Input triangle mesh data")
#ifdef MEGAMOL_NG_MESH
					, ngmesh_input_slot("ngmesh_data", "Input triangle NG mesh data")
#endif
				{
					// Create file name textbox
					this->filename_slot << new core::param::FilePathParam("");
					this->MakeSlotAvailable(&this->filename_slot);

					// Create enum for ASCII/binary option
					this->ascii_binary_slot << new core::param::EnumParam(0);
					this->ascii_binary_slot.Param<core::param::EnumParam>()->SetTypePair(0, "Binary");
					this->ascii_binary_slot.Param<core::param::EnumParam>()->SetTypePair(1, "ASCII");
					this->MakeSlotAvailable(&this->ascii_binary_slot);

					// Create input slot for triangle mesh data
					this->mesh_input_slot.SetCompatibleCall<geocalls::CallTriMeshDataDescription>();
					this->MakeSlotAvailable(&this->mesh_input_slot);

#ifdef MEGAMOL_NG_MESH
					// Create input slot for triangle mesh data
					this->ngmesh_input_slot.SetCompatibleCall<ngmesh::CallNGMeshRenderBatchesDescription>();
					this->MakeSlotAvailable(&this->ngmesh_input_slot);
#endif
				}

				STLWriter::~STLWriter()
				{ }

				bool STLWriter::create()
				{
					return true;
				}

				bool STLWriter::getCapabilities(core::DataWriterCtrlCall& call)
				{
					call.SetAbortable(false);

					return true;
				}

				bool STLWriter::run()
				{
					// Get frame
					core::AbstractGetData3DCall* call;

					auto* mesh_call = this->mesh_input_slot.CallAs<geocalls::CallTriMeshData>();

#ifdef MEGAMOL_NG_MESH
					auto* ngmesh_call = this->ngmesh_input_slot.CallAs<ngmesh::CallNGMeshRenderBatches>();

					if (mesh_call == nullptr && ngmesh_call == nullptr)
					{
						vislib::sys::Log::DefaultLog.WriteError("No data source connected");

						return false;
					}
					else if (mesh_call != nullptr && ngmesh_call != nullptr)
					{
						vislib::sys::Log::DefaultLog.WriteError("Two data sources connected");

						return false;
					}
					else if (mesh_call != nullptr)
					{
						call = mesh_call;
					}
					else
					{
						call = ngmesh_call;
					}
#else
					call = dynamic_cast<core::AbstractGetData3DCall*>(mesh_call);

					if (call == nullptr)
					{
						vislib::sys::Log::DefaultLog.WriteError("No data source connected");

						return false;
					}
#endif

					// Call for get extents
					call->SetFrameID(0);

					if (!(*call)(1))
					{
						return false;
					}

					// Call for get data
					if (!(*call)(0))
					{
						return false;
					}

					// Get data and save it to file
					const auto& vislib_filename = this->filename_slot.Param<core::param::FilePathParam>()->Value();
					const std::string filename(vislib_filename.PeekBuffer());

					try
					{
#ifdef MEGAMOL_NG_MESH
						if (mesh_call != nullptr)
						{
							if (this->ascii_binary_slot.Param<core::param::EnumParam>()->Value() == 0)
							{
								write_binary(filename, get_mesh_data(*mesh_call));
							}
							else
							{
								write_ascii(filename, get_mesh_data(*mesh_call));
							}
						}
						else
						{
							if (this->ascii_binary_slot.Param<core::param::EnumParam>()->Value() == 0)
							{
								write_binary(filename, get_ngmesh_data(*ngmesh_call));
							}
							else
							{
								write_ascii(filename, get_ngmesh_data(*ngmesh_call));
							}
						}
#else
						if (this->ascii_binary_slot.Param<core::param::EnumParam>()->Value() == 0)
						{
							write_binary(filename, get_mesh_data(*mesh_call));
						}
						else
						{
							write_ascii(filename, get_mesh_data(*mesh_call));
						}
#endif
					}
					catch (const std::runtime_error& ex)
					{
						vislib::sys::Log::DefaultLog.WriteError("Request for writing to STL file failed: %s", ex.what());

						return false;
					}

					return true;
				}

				void STLWriter::release()
				{ }

				std::tuple<std::size_t, const float*, const float*> STLWriter::get_mesh_data(geocalls::CallTriMeshData& call) const
				{
					return std::make_tuple(static_cast<std::size_t>(call.Objects()->GetTriCount()),
						call.Objects()->GetVertexPointerFloat(), call.Objects()->GetNormalPointerFloat());
				}

#ifdef MEGAMOL_NG_MESH
				std::tuple<std::size_t, const float*, const float*> STLWriter::get_ngmesh_data(ngmesh::CallNGMeshRenderBatches& call) const
				{
					return std::make_tuple(static_cast<std::size_t>(0), nullptr, nullptr); // TODO
				}
#endif

				void STLWriter::write_binary(const std::string& filename, const std::tuple<std::size_t, const float*, const float*>& mesh) const
				{
					// Open or create file
					std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);

					if (ofs.good())
					{
						// Write header
						std::string header_message("MegaMol by University of Stuttgart, Germany");

						std::vector<char> header_buffer(header_message.begin(), header_message.end());
						header_buffer.resize(80);

						ofs.write(header_buffer.data(), header_buffer.size());

						// Write number of triangles
						uint32_t num_triangles = static_cast<uint32_t>(std::get<0>(mesh));

						ofs.write(reinterpret_cast<char*>(&num_triangles), sizeof(uint32_t));

						// Write vertices and normals
						const uint16_t additional_attribute = 42;

						for (std::size_t triangle_index = 0; triangle_index < std::get<0>(mesh); ++triangle_index)
						{
							ofs.write(reinterpret_cast<const char*>(&std::get<2>(mesh)[triangle_index * 9]), 3 * sizeof(float));
							ofs.write(reinterpret_cast<const char*>(&std::get<1>(mesh)[triangle_index * 9]), 9 * sizeof(float));
							ofs.write(reinterpret_cast<const char*>(&additional_attribute), sizeof(uint16_t));
						}

						ofs.close();
					}
					else
					{
						std::stringstream ss;
						ss << "Binary STL file '" << filename << "' could not be written";

						throw std::runtime_error(ss.str());
					}
				}

				void STLWriter::write_ascii(const std::string& filename, const std::tuple<std::size_t, const float*, const float*>& mesh) const
				{
					// Open or create file
					std::ofstream ofs(filename, std::ios_base::out);

					if (ofs.good())
					{
						// Write mesh
						ofs << "solid megamol_mesh\n";
						ofs << std::scientific;

						for (std::size_t triangle_index = 0; triangle_index < std::get<0>(mesh); ++triangle_index)
						{
							ofs << "\tfacet\n";

							ofs << "\t\tnormal "
								<< std::get<2>(mesh)[triangle_index * 9 + 0] << " "
								<< std::get<2>(mesh)[triangle_index * 9 + 1] << " "
								<< std::get<2>(mesh)[triangle_index * 9 + 2] << "\n";

							ofs << "\t\touter loop\n";

							ofs << "\t\t\tvertex "
								<< std::get<1>(mesh)[triangle_index * 9 + 0] << " "
								<< std::get<1>(mesh)[triangle_index * 9 + 1] << " "
								<< std::get<1>(mesh)[triangle_index * 9 + 2] << "\n";
							ofs << "\t\t\tvertex "
								<< std::get<1>(mesh)[triangle_index * 9 + 3] << " "
								<< std::get<1>(mesh)[triangle_index * 9 + 4] << " "
								<< std::get<1>(mesh)[triangle_index * 9 + 5] << "\n";
							ofs << "\t\t\tvertex "
								<< std::get<1>(mesh)[triangle_index * 9 + 6] << " "
								<< std::get<1>(mesh)[triangle_index * 9 + 7] << " "
								<< std::get<1>(mesh)[triangle_index * 9 + 8] << "\n";

							ofs << "\t\tendloop\n";

							ofs << "\tendfacet" << std::endl;
						}

						ofs << "endsolid megamol_mesh" << std::flush;

						ofs.close();
					}
					else
					{
						std::stringstream ss;
						ss << "ASCII STL file '" << filename << "' could not be written";

						throw std::runtime_error(ss.str());
					}
				}
			}
		}
	}
}