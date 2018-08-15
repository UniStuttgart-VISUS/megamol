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

#include <fstream>
#include <numeric>
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
					auto* call = this->mesh_input_slot.CallAs<geocalls::CallTriMeshData>();

					if (call == nullptr)
					{
						vislib::sys::Log::DefaultLog.WriteError("No data source connected");

						return false;
					}

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
						const std::size_t num_triangles = static_cast<std::size_t>(call->Objects()->GetTriCount());

						if (this->ascii_binary_slot.Param<core::param::EnumParam>()->Value() == 0)
						{
							// Sanity check
							if (call->Objects()->GetVertexDataType() == geocalls::CallTriMeshData::Mesh::DT_NONE)
							{
								throw std::runtime_error("Illegal vertex data type");
							}
							if (call->Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_NONE)
							{
								throw std::runtime_error("Illegal normal data type");
							}

							// Convert vertices or normals to float if necessary
							pointer_wrapper<float, std::vector<float>> vertices, normals;

							if (call->Objects()->GetVertexDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT)
							{
								vertices = convert_if_necessary<float>(call->Objects()->GetVertexPointerFloat(), num_triangles * 9);
							}
							else
							{
								vertices = convert_if_necessary<float>(call->Objects()->GetVertexPointerDouble(), num_triangles * 9);
							}

							if (call->Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT)
							{
								normals = convert_if_necessary<float>(call->Objects()->GetNormalPointerFloat(), num_triangles * 9);
							}
							else
							{
								normals = convert_if_necessary<float>(call->Objects()->GetNormalPointerDouble(), num_triangles * 9);
							}

							// Write binary file
							if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE)
							{
								write_binary(filename, static_cast<uint32_t>(num_triangles), vertices.get(), normals.get(), call->Objects()->GetTriIndexPointerByte());
							}
							else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16)
							{
								write_binary(filename, static_cast<uint32_t>(num_triangles), vertices.get(), normals.get(), call->Objects()->GetTriIndexPointerUInt16());
							}
							else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32)
							{
								write_binary(filename, static_cast<uint32_t>(num_triangles), vertices.get(), normals.get(), call->Objects()->GetTriIndexPointerUInt32());
							}
							else
							{
								std::vector<unsigned int> indices(num_triangles * 3);
								std::iota(indices.begin(), indices.end(), 0);

								write_binary(filename, static_cast<uint32_t>(num_triangles), vertices.get(), normals.get(), indices.data());
							}
						}
						else
						{
							// Sanity check
							if (call->Objects()->GetVertexDataType() == geocalls::CallTriMeshData::Mesh::DT_NONE)
							{
								throw std::runtime_error("Illegal vertex data type");
							}
							if (call->Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_NONE)
							{
								throw std::runtime_error("Illegal normal data type");
							}
							
							// Write ASCII file
							if (call->Objects()->GetVertexDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT)
							{
								if (call->Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT)
								{
									if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerFloat(), call->Objects()->GetTriIndexPointerByte());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerFloat(), call->Objects()->GetTriIndexPointerUInt16());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerFloat(), call->Objects()->GetTriIndexPointerUInt32());
									}
									else
									{
										std::vector<unsigned int> indices(num_triangles * 3);
										std::iota(indices.begin(), indices.end(), 0);

										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerFloat(), indices.data());
									}
								}
								else
								{
									if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerDouble(), call->Objects()->GetTriIndexPointerByte());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerDouble(), call->Objects()->GetTriIndexPointerUInt16());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerDouble(), call->Objects()->GetTriIndexPointerUInt32());
									}
									else
									{
										std::vector<unsigned int> indices(num_triangles * 3);
										std::iota(indices.begin(), indices.end(), 0);

										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerFloat(),
											call->Objects()->GetNormalPointerDouble(), indices.data());
									}
								}
							}
							else
							{
								if (call->Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT)
								{
									if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerFloat(), call->Objects()->GetTriIndexPointerByte());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerFloat(), call->Objects()->GetTriIndexPointerUInt16());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerFloat(), call->Objects()->GetTriIndexPointerUInt32());
									}
									else
									{
										std::vector<unsigned int> indices(num_triangles * 3);
										std::iota(indices.begin(), indices.end(), 0);

										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerFloat(), indices.data());
									}
								}
								else
								{
									if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerDouble(), call->Objects()->GetTriIndexPointerByte());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerDouble(), call->Objects()->GetTriIndexPointerUInt16());
									}
									else if (call->Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32)
									{
										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerDouble(), call->Objects()->GetTriIndexPointerUInt32());
									}
									else
									{
										std::vector<unsigned int> indices(num_triangles * 3);
										std::iota(indices.begin(), indices.end(), 0);

										write_ascii(filename, num_triangles, call->Objects()->GetVertexPointerDouble(),
											call->Objects()->GetNormalPointerDouble(), indices.data());
									}
								}
							}
						}
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

				template <typename IT>
				void STLWriter::write_binary(const std::string& filename, const uint32_t num_triangles, const float* vertices, const float* normals, const IT* indices) const
				{
					static_assert(std::is_integral<IT>::value, "Indices must be of integral type");

					if (num_triangles == 0)
					{
						vislib::sys::Log::DefaultLog.WriteWarn("Cannot write STL file. Number of triangles is zero!");
						return;
					}

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
						ofs.write(reinterpret_cast<const char*>(&num_triangles), sizeof(uint32_t));

						// Write vertices and normals
						const uint16_t additional_attribute = 21313;

						for (uint32_t triangle_index = 0; triangle_index < num_triangles; ++triangle_index)
						{
							ofs.write(reinterpret_cast<const char*>(&normals[indices[triangle_index * 3] * 3]), 3 * sizeof(float));
							ofs.write(reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 0] * 3]), 3 * sizeof(float));
							ofs.write(reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 1] * 3]), 3 * sizeof(float));
							ofs.write(reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 2] * 3]), 3 * sizeof(float));
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

				template <typename VFT, typename NFT, typename IT>
				void STLWriter::write_ascii(const std::string& filename, const std::size_t num_triangles, const VFT* vertices, const NFT* normals, const IT* indices) const
				{
					static_assert(std::is_floating_point<VFT>::value, "Vertices must be of floating point type");
					static_assert(std::is_floating_point<NFT>::value, "Normals must be of floating point type");
					static_assert(std::is_integral<IT>::value, "Indices must be of integral type");

					if (num_triangles == 0)
					{
						vislib::sys::Log::DefaultLog.WriteWarn("Cannot write STL file. Number of triangles is zero!");
						return;
					}

					// Open or create file
					std::ofstream ofs(filename, std::ios_base::out);

					if (ofs.good())
					{
						// Write mesh
						ofs << "solid megamol_mesh\n";
						ofs << std::scientific;

						for (std::size_t triangle_index = 0; triangle_index < num_triangles; ++triangle_index)
						{
							ofs << "\tfacet\n";

							ofs << "\t\tnormal "
								<< normals[indices[triangle_index * 3] * 3 + 0] << " "
								<< normals[indices[triangle_index * 3] * 3 + 1] << " "
								<< normals[indices[triangle_index * 3] * 3 + 2] << "\n";

							ofs << "\t\touter loop\n";

							ofs << "\t\t\tvertex "
								<< vertices[indices[triangle_index * 3 + 0] * 3 + 0] << " "
								<< vertices[indices[triangle_index * 3 + 0] * 3 + 1] << " "
								<< vertices[indices[triangle_index * 3 + 0] * 3 + 2] << "\n";
							ofs << "\t\t\tvertex "
								<< vertices[indices[triangle_index * 3 + 1] * 3 + 0] << " "
								<< vertices[indices[triangle_index * 3 + 1] * 3 + 1] << " "
								<< vertices[indices[triangle_index * 3 + 1] * 3 + 2] << "\n";
							ofs << "\t\t\tvertex "
								<< vertices[indices[triangle_index * 3 + 2] * 3 + 0] << " "
								<< vertices[indices[triangle_index * 3 + 2] * 3 + 1] << " "
								<< vertices[indices[triangle_index * 3 + 2] * 3 + 2] << "\n";

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