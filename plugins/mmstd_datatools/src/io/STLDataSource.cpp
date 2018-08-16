/*
* STLDataSource.cpp
*
* Copyright (C) 2018 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "../stdafx.h"
#include "STLDataSource.h"

#include "mmcore/Call.h"
#include "mmcore/AbstractGetData3DCall.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/DataHash.h"

#include "geometry_calls/CallTriMeshData.h"

#ifdef MEGAMOL_NG_MESH
#include "ng_mesh/CallNGMeshRenderBatches.h"
#endif

#include "vislib/sys/Log.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace megamol
{
	namespace stdplugin
	{
		namespace datatools
		{
			namespace io
			{
				STLDataSource::STLDataSource() : filename_slot("STL file", "The name of to the STL file to load")
					, mesh_output_slot("mesh_data", "Slot to request mesh data")
#ifdef MEGAMOL_NG_MESH
					, ngmesh_output_slot("ngmesh_data", "Slot to request mesh data for the NGMeshRenderer")
					, batch_data(std::make_unique<ngmesh::CallNGMeshRenderBatches::RenderBatchesData>())
#endif
				{
					// Create file name textbox
					this->filename_slot << new core::param::FilePathParam("");
					Module::MakeSlotAvailable(&this->filename_slot);

					// Create output slot for triangle mesh data
					this->mesh_output_slot.SetCallback(geocalls::CallTriMeshData::ClassName(), "GetExtent", &STLDataSource::get_extent_callback);
					this->mesh_output_slot.SetCallback(geocalls::CallTriMeshData::ClassName(), "GetData", &STLDataSource::get_mesh_data_callback);

					Module::MakeSlotAvailable(&this->mesh_output_slot);

					// Create output slot for ng mesh data
#ifdef MEGAMOL_NG_MESH
					this->shader_filename_slot << new core::param::FilePathParam("stl_triangles");
					Module::MakeSlotAvailable(&this->shader_filename_slot);

					this->ngmesh_output_slot.SetCallback<STLDataSource>("CallNGMeshRenderBatches", "GetExtent", &STLDataSource::get_extent_callback);
					this->ngmesh_output_slot.SetCallback<STLDataSource>("CallNGMeshRenderBatches", "GetData", &STLDataSource::get_ngmesh_data_callback);

					Module::MakeSlotAvailable(&this->ngmesh_output_slot);
#endif
				}

				STLDataSource::~STLDataSource()
				{ }

				bool STLDataSource::create()
				{
					return true;
				}

				bool STLDataSource::get_extent_callback(core::Call& caller)
				{
					// Get mesh call
					auto& call = dynamic_cast<core::AbstractGetData3DCall&>(caller);

					if (this->filename_slot.IsDirty())
					{
						this->filename_slot.ResetDirty();

						// Read data
						const auto& vislib_filename = this->filename_slot.Param<core::param::FilePathParam>()->Value();
						const std::string filename(vislib_filename.PeekBuffer());

						try
						{
							read(filename);
						}
						catch (const std::runtime_error& ex)
						{
							vislib::sys::Log::DefaultLog.WriteError("Request for extent of plugin 'STL Reader' failed: %s", ex.what());

							return false;
						}

						// Extract extent information
						this->min_x = this->min_y = this->min_z = std::numeric_limits<float>::max();
						this->max_x = this->max_y = this->max_z = std::numeric_limits<float>::lowest();

						const float* vertices = reinterpret_cast<float*>(&this->vertex_normal_buffer[2 * sizeof(uint32_t)]);

						for (std::size_t value_index = 0; value_index < static_cast<std::size_t>(this->num_triangles * 9); )
						{
							const float x = vertices[value_index++];
							const float y = vertices[value_index++];
							const float z = vertices[value_index++];

							this->min_x = std::min(this->min_x, x);
							this->min_y = std::min(this->min_y, y);
							this->min_z = std::min(this->min_z, z);

							this->max_x = std::max(this->max_x, x);
							this->max_y = std::max(this->max_y, y);
							this->max_z = std::max(this->max_z, z);
						}

						vislib::sys::Log::DefaultLog.WriteInfo("Extent: [%.2f, %.2f, %.2f] x [%.2f, %.2f, %.2f]",
							this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z);
					}

					call.SetExtent(static_cast<unsigned int>(0), this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z);

					return true;
				}

				bool STLDataSource::get_mesh_data_callback(core::Call& caller)
				{
					// Get mesh call
					auto& call = dynamic_cast<geocalls::CallTriMeshData&>(caller);

					if (call.DataHash() != static_cast<SIZE_T>(hash()))
					{
						// Read data if necessary
						if (this->filename_slot.IsDirty())
						{
							if (!get_extent_callback(caller))
							{
								return false;
							}
						}
						
						call.SetDataHash(static_cast<SIZE_T>(hash()));

						// Fill call
						this->mesh.SetVertexData(static_cast<unsigned int>(this->num_triangles * 3),
							reinterpret_cast<float*>(&this->vertex_normal_buffer.data()[2 * sizeof(uint32_t)]),
							reinterpret_cast<float*>(&this->vertex_normal_buffer.data()[2 * sizeof(uint32_t) + 9 * this->num_triangles * sizeof(float)]),
							nullptr, nullptr, false);

						this->mesh.SetTriangleData(static_cast<unsigned int>(this->num_triangles), this->index_buffer.data(), false);
						
						call.SetObjects(1, &this->mesh);
					}
					
					return true;
				}

#ifdef MEGAMOL_NG_MESH
				bool STLDataSource::get_ngmesh_data_callback(core::Call& caller)
				{
					// Get ng mesh call
					auto& call = dynamic_cast<ngmesh::CallNGMeshRenderBatches&>(caller);

					if (this->filename_slot.IsDirty())
					{
						// Read data if necessary
						if (call.DataHash() != static_cast<SIZE_T>(hash()))
						{
							if (!get_extent_callback(caller))
							{
								return false;
							}
						}

						// Create new batch data
						this->batch_data = std::make_unique<ngmesh::CallNGMeshRenderBatches::RenderBatchesData>();
						call.setRenderBatches(this->batch_data.get());

						// Create shader program
						ngmesh::ShaderPrgmDataAccessor shader_program;

						const auto& shader_filename_vislib = this->shader_filename_slot.Param<core::param::FilePathParam>()->Value();
						const std::string shader_filename(shader_filename_vislib.PeekBuffer());
						std::vector<char> shader_filename_vec(shader_filename.begin(), shader_filename.end());
						shader_filename_vec.push_back('\0');

						shader_program.char_cnt = shader_filename_vec.size();
						shader_program.raw_string = shader_filename_vec.data();

						// Store mesh data
						ngmesh::MeshDataAccessor mesh_data;

						mesh_data.vertex_data.byte_size = this->vertex_normal_buffer.size();
						mesh_data.vertex_data.raw_data = this->vertex_normal_buffer.data();
						mesh_data.vertex_data.buffer_cnt = 2;

						mesh_data.index_data.byte_size = this->index_buffer.size() * sizeof(unsigned int);
						mesh_data.index_data.raw_data = reinterpret_cast<uint8_t*>(this->index_buffer.data());
						mesh_data.index_data.index_type = GL_UNSIGNED_INT;

						std::array<ngmesh::MeshDataAccessor::VertexLayoutData::Attribute, 2> attributes;

						attributes[0].size = 3;
						attributes[0].type = GL_FLOAT;
						attributes[0].normalized = GL_FALSE;
						attributes[0].offset = 0;

						attributes[1].size = 3;
						attributes[1].type = GL_FLOAT;
						attributes[1].normalized = GL_TRUE;
						attributes[1].offset = 0;

						mesh_data.vertex_descriptor.attribute_cnt = 2;
						mesh_data.vertex_descriptor.attributes = attributes.data();
						mesh_data.vertex_descriptor.stride = 0;

						mesh_data.usage = GL_STATIC_DRAW;
						mesh_data.primitive_type = GL_TRIANGLES;

						// Set draw commands
						ngmesh::DrawCommandDataAccessor draw_commands;
						ngmesh::DrawCommandDataAccessor::DrawElementsCommand draw_commands_data;

						draw_commands.draw_cnt = 1;
						draw_commands.data = &draw_commands_data;

						draw_commands_data.cnt = static_cast<GLuint>(this->vertex_buffer.size() / 3);
						draw_commands_data.instance_cnt = 1;
						draw_commands_data.first_idx = 0;
						draw_commands_data.base_vertex = 0;
						draw_commands_data.base_instance = 0;

						// Set object shader parameters
						ngmesh::ObjectShaderParamsDataAccessor obj_shader_params;

						obj_shader_params.byte_size = 0;
						obj_shader_params.raw_data = nullptr;

						// Set material shader parameters
						ngmesh::MaterialShaderParamsDataAccessor mtl_shader_params;

						mtl_shader_params.elements_cnt = 0;
						mtl_shader_params.data = nullptr;

						// Store information and data in batch
						this->batch_data->addBatch(shader_program, mesh_data, draw_commands, obj_shader_params, mtl_shader_params);
					}

					this->filename_slot.ResetDirty();

					return true;
				}
#endif

				void STLDataSource::release()
				{ }

				void STLDataSource::read(const std::string& filename)
				{
					std::ifstream ifs(filename, std::ofstream::in | std::ofstream::binary);

					if (ifs.good())
					{
						// Read first non-white characters to identify an ASCII file
						std::string line;

						while (line.empty() && !ifs.eof())
						{
							// Read and convert line to lower case
							std::getline(ifs, line);
							std::transform(line.begin(), line.end(), line.begin(), ::tolower);

							line.erase(0, line.find_first_not_of("\f\n\r\t\v "));
						}

						// Read file
						if (line.substr(0, line.find_first_of("\f\n\r\t\v ")).compare("solid") == 0)
						{
							read_ascii(filename);
						}
						else
						{
							read_binary(filename);
						}
					}
					else
					{
						std::stringstream ss;
						ss << "STL file '" << filename << "' not found or inaccessible";

						throw std::runtime_error(ss.str());
					}
				}

				void STLDataSource::read_binary(const std::string& filename)
				{
					std::ifstream ifs(filename, std::ofstream::in | std::ofstream::binary);

					if (ifs.good())
					{
						// Get number of triangles from header
						ifs.ignore(80 * sizeof(uint8_t));
						ifs.read(reinterpret_cast<char*>(&this->num_triangles), sizeof(uint32_t));

						// Sanity check for file size
						ifs.ignore(std::numeric_limits<std::streamsize>::max());
						std::streamsize file_size = ifs.gcount();

						if (file_size != this->num_triangles * 50)
						{
							throw std::runtime_error("File size does not match the number of triangles.");
						}

						vislib::sys::Log::DefaultLog.WriteInfo("Number of triangles from binary STL file: %d", this->num_triangles);

						// Read data
						ifs.seekg(80 * sizeof(uint8_t) + sizeof(uint32_t));

						const std::size_t header_block_size = sizeof(uint32_t);
						const std::size_t data_block_size = 9 * this->num_triangles * sizeof(float);

						const std::size_t data_offset_1 = 2 * header_block_size;
						const std::size_t data_offset_2 = 2 * header_block_size + data_block_size;

						this->vertex_normal_buffer.resize(2 * data_block_size + 2 * header_block_size);

						reinterpret_cast<uint32_t&>(this->vertex_normal_buffer[0 * header_block_size]) = static_cast<uint32_t>(data_offset_1);
						reinterpret_cast<uint32_t&>(this->vertex_normal_buffer[1 * header_block_size]) = static_cast<uint32_t>(data_offset_2);

						std::size_t vertex_position = data_offset_1;
						std::size_t normal_position = data_offset_2;

						for (std::size_t triangle_index = 0; triangle_index < static_cast<std::size_t>(this->num_triangles); ++triangle_index)
						{
							ifs.read(reinterpret_cast<char*>(&this->vertex_normal_buffer[normal_position]), 3 * sizeof(float));
							ifs.read(reinterpret_cast<char*>(&this->vertex_normal_buffer[vertex_position]), 9 * sizeof(float));
							ifs.ignore(sizeof(uint16_t));

							std::memcpy(&this->vertex_normal_buffer[normal_position + 3 * sizeof(float)], &this->vertex_normal_buffer[normal_position], 3 * sizeof(float));
							std::memcpy(&this->vertex_normal_buffer[normal_position + 6 * sizeof(float)], &this->vertex_normal_buffer[normal_position], 3 * sizeof(float));

							vertex_position += 9 * sizeof(float);
							normal_position += 9 * sizeof(float);
						}

						// Fill index buffer
						this->index_buffer.resize(this->num_triangles * 3);

						std::iota(this->index_buffer.begin(), this->index_buffer.end(), 0);
					}
					else
					{
						std::stringstream ss;
						ss << "STL file '" << filename << "' not found or inaccessible";

						throw std::runtime_error(ss.str());
					}
				}

				void STLDataSource::read_ascii(const std::string& filename)
				{
					std::ifstream ifs(filename, std::ofstream::in);

					if (ifs.good())
					{
						// Get file size and estimate number of triangles
						ifs.ignore(std::numeric_limits<std::streamsize>::max());
						std::streamsize file_size = ifs.gcount();
						ifs.seekg(ifs.beg);

						const std::size_t estimated_num_triangles = static_cast<std::size_t>(1.5f * (static_cast<float>(file_size) / 256.0f));

						std::vector<float> vertices, normals;
						vertices.reserve(9 * estimated_num_triangles);
						normals.reserve(3 * estimated_num_triangles);

						// Parse file
						parser_states_t state = EXPECT_SOLID;

						std::string line;
						std::string name;

						while (!ifs.eof())
						{
							// Read and convert line to lower case
							std::getline(ifs, line);
							std::transform(line.begin(), line.end(), line.begin(), ::tolower);

							line.erase(0, line.find_first_not_of("\f\n\r\t\v "));

							while (!line.empty())
							{
								// Extract word from line
								const std::string word = line.substr(0, line.find_first_of("\f\n\r\t\v "));

								line.erase(0, word.size());
								line.erase(0, line.find_first_not_of("\f\n\r\t\v "));

								// Parse word
								switch (state)
								{
								case EXPECT_SOLID:
									if (word.compare("solid") == 0)
									{
										state = EXPECT_NAME;
									}
									else
									{
										throw std::runtime_error("Expected ASCII STL file to begin with keyword 'solid'");
									}

									break;
								case EXPECT_NAME:
									name = word;

									state = EXPECT_FACET_OR_ENDSOLID;

									break;
								case EXPECT_FACET_OR_ENDSOLID:
									if (word.compare("facet") == 0)
									{
										state = EXPECT_NORMAL;
									}
									else if (word.compare("endsolid") == 0)
									{
										state = EXPECT_ENDNAME;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'facet' or 'endsolid' after solid name or 'endfacet' in ASCII STL file");
									}

									break;
								case EXPECT_NORMAL:
									if (word.compare("normal") == 0)
									{
										state = EXPECT_NORMAL_I;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'normal' after keyword 'facet' in ASCII STL file");
									}

									break;
								case EXPECT_NORMAL_I:
									normals.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_NORMAL_J;

									break;
								case EXPECT_NORMAL_J:
									normals.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_NORMAL_K;

									break;
								case EXPECT_NORMAL_K:
									normals.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_OUTER;

									break;
								case EXPECT_OUTER:
									if (word.compare("outer") == 0)
									{
										state = EXPECT_LOOP;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'outer' after the normal values in ASCII STL file");
									}

									break;
								case EXPECT_LOOP:
									if (word.compare("loop") == 0)
									{
										state = EXPECT_VERTEX_1;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'loop' after keyword 'outer' in ASCII STL file");
									}

									break;
								case EXPECT_VERTEX_1:
									if (word.compare("vertex") == 0)
									{
										state = EXPECT_VERTEX_1_X;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'vertex' after keyword 'loop' in ASCII STL file");
									}

									break;
								case EXPECT_VERTEX_1_X:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_1_Y;

									break;
								case EXPECT_VERTEX_1_Y:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_1_Z;

									break;
								case EXPECT_VERTEX_1_Z:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_2;

									break;
								case EXPECT_VERTEX_2:
									if (word.compare("vertex") == 0)
									{
										state = EXPECT_VERTEX_2_X;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'vertex' after the first vertex values in ASCII STL file");
									}

									break;
								case EXPECT_VERTEX_2_X:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_2_Y;

									break;
								case EXPECT_VERTEX_2_Y:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_2_Z;

									break;
								case EXPECT_VERTEX_2_Z:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_3;

									break;
								case EXPECT_VERTEX_3:
									if (word.compare("vertex") == 0)
									{
										state = EXPECT_VERTEX_3_X;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'vertex' after the second vertex values in ASCII STL file");
									}

									break;
								case EXPECT_VERTEX_3_X:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_3_Y;

									break;
								case EXPECT_VERTEX_3_Y:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_VERTEX_3_Z;

									break;
								case EXPECT_VERTEX_3_Z:
									vertices.push_back(static_cast<float>(atof(word.c_str())));

									state = EXPECT_ENDLOOP;

									break;
								case EXPECT_ENDLOOP:
									if (word.compare("endloop") == 0)
									{
										state = EXPECT_ENDFACET;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'endloop' after the third vertex values in ASCII STL file");
									}

									break;
								case EXPECT_ENDFACET:
									if (word.compare("endfacet") == 0)
									{
										state = EXPECT_FACET_OR_ENDSOLID;
									}
									else
									{
										throw std::runtime_error("Expected keyword 'endfacet' after keyword 'endloop' in ASCII STL file");
									}

									break;
								case EXPECT_ENDNAME:
									if (word.compare(name) == 0)
									{
										state = EXPECT_EOF;
									}
									else
									{
										throw std::runtime_error("Expected same solid name after keyword 'endsolid' in ASCII STL file");
									}

									break;
								case EXPECT_EOF:
								default:
									vislib::sys::Log::DefaultLog.WriteWarn("Found more text after keyword 'endsolid' in ASCII STL file. %s",
										"Maybe more than one object is stored in the file. This is not supported by this reader.");
								}
							}
						}

						this->num_triangles = static_cast<uint32_t>(vertices.size() / 9);
						vislib::sys::Log::DefaultLog.WriteInfo("Number of triangles from ASCII STL file: %d", this->num_triangles);

						// Fill buffer
						this->vertex_normal_buffer.resize(2 * sizeof(uint32_t) + 18 * this->num_triangles * sizeof(float));

						reinterpret_cast<uint32_t*>(this->vertex_normal_buffer.data())[0] = 2 * sizeof(uint32_t);
						reinterpret_cast<uint32_t*>(this->vertex_normal_buffer.data())[1] = 2 * sizeof(uint32_t) + 9 * this->num_triangles * sizeof(float);

						std::memcpy(&this->vertex_normal_buffer[2 * sizeof(uint32_t)], vertices.data(), 9 * this->num_triangles * sizeof(float));

						const std::size_t offset = 2 * sizeof(uint32_t) + 9 * this->num_triangles * sizeof(float);
						std::size_t position = 0;

						for (std::size_t triangle_index = 0; triangle_index < this->num_triangles; ++triangle_index)
						{
							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 0];
							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 1];
							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 2];

							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 0];
							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 1];
							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 2];

							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 0];
							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 1];
							reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] = normals[triangle_index * 3 + 2];
						}

						// Fill index buffer
						this->index_buffer.resize(this->num_triangles * 3);

						std::iota(this->index_buffer.begin(), this->index_buffer.end(), 0);
					}
				}

				uint32_t STLDataSource::hash() const
				{
					if (this->vertex_normal_buffer.empty())
					{
						return 0;
					}

					const uint32_t first_dataset = reinterpret_cast<const uint32_t&>(this->vertex_normal_buffer[0 * sizeof(uint32_t)]);
					const uint32_t second_dataset = reinterpret_cast<const uint32_t&>(this->vertex_normal_buffer[1 * sizeof(uint32_t)]);

					return core::utility::DataHash(
						// Header
						first_dataset,
						second_dataset,
						// First vertex
						reinterpret_cast<const float&>(this->vertex_normal_buffer[first_dataset + 0 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[first_dataset + 1 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[first_dataset + 2 * sizeof(float)]),
						// Last vertex
						reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset - 3 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset - 2 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset - 1 * sizeof(float)]),
						// First normal
						reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset + 0 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset + 1 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset + 2 * sizeof(float)]),
						// Last normal
						reinterpret_cast<const float&>(this->vertex_normal_buffer[2 * second_dataset - first_dataset - 3 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[2 * second_dataset - first_dataset - 2 * sizeof(float)]),
						reinterpret_cast<const float&>(this->vertex_normal_buffer[2 * second_dataset - first_dataset - 1 * sizeof(float)])
					);
				}
			}
		}
	}
}