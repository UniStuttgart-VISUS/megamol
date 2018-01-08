/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include <iostream>
#include <random>
#include <string>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "stdafx.h"
#include "ArchVisMSMDataSource.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "external/tiny_gltf.h"

using namespace megamol::archvis;
using namespace megamol::ngmesh;

ArchVisMSMDataSource::ArchVisMSMDataSource() :
	m_shaderFilename_slot("Shader", "The name of to the shader file to load"),
	m_partsList_slot("Parts list", "The path to the parts list file to load"),
	m_nodeElement_table_slot("Node/Element table", "The path to the node/element table to load"),
	m_IPAdress_slot("Ip adress", "The ip adress of the sensor data transfer")
{
	this->m_shaderFilename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_shaderFilename_slot);

	this->m_partsList_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_partsList_slot);

	m_nodeElement_table_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_nodeElement_table_slot);
	
	m_IPAdress_slot << new core::param::IntParam(0);
	this->MakeSlotAvailable(&this->m_IPAdress_slot);
}

ArchVisMSMDataSource::~ArchVisMSMDataSource()
{
}

bool ArchVisMSMDataSource::getDataCallback(megamol::core::Call& caller)
{
	CallNGMeshRenderBatches* render_batches_call = dynamic_cast<CallNGMeshRenderBatches*>(&caller);
	if (render_batches_call == NULL)
		return false;
	if (this->m_partsList_slot.IsDirty() ||
		this->m_shaderFilename_slot.IsDirty() ||
		this->m_nodeElement_table_slot.IsDirty() )
	{
		// TODO handle different slots seperatly ?
		this->m_shaderFilename_slot.ResetDirty();
		this->m_partsList_slot.ResetDirty();
		this->m_nodeElement_table_slot.ResetDirty();

		// Clear render batches TODO: add explicit clear function?
		CallNGMeshRenderBatches::RenderBatchesData empty_render_batches;
		m_render_batches = empty_render_batches;

		m_bbox.Set(-10.0f, -10.0f, -10.0f, 10.0f, 10.0f, 10.0f);

		auto vislib_shader_filename = m_shaderFilename_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string shdr_filename(vislib_shader_filename.PeekBuffer());

		auto vislib_partsList_filename = m_partsList_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string partsList_filename(vislib_partsList_filename.PeekBuffer());

		auto vislib_nodesElement_filename = m_nodeElement_table_slot.Param<megamol::core::param::FilePathParam>()->Value();
		std::string nodesElement_filename(vislib_nodesElement_filename.PeekBuffer());

		load(shdr_filename, partsList_filename, nodesElement_filename);
	}

	// TODO handle IP slot is dirty

	render_batches_call->setRenderBatches(&m_render_batches);

	return true;
}

std::vector<std::string> ArchVisMSMDataSource::parsePartsList(std::string const& filename)
{
	std::vector<std::string> parts;

	std::ifstream file;
	file.open(filename, std::ifstream::in);

	if (file.is_open())
	{
		file.seekg(0, std::ifstream::beg);

		while (!file.eof())
		{
			parts.push_back(std::string());

			getline(file, parts.back(), '\n');
		}
	}

	return parts;
}

void ArchVisMSMDataSource::parseNodeElementTable(
	std::string const& filename,
	std::vector<Node>& nodes,
	std::vector<FloorElement>& floor_elements,
	std::vector<BeamElement>& beam_elements,
	std::vector<DiagonalElement>& diagonal_elements)
{
	std::ifstream file;
	file.open(filename, std::ifstream::in);

	if (file.is_open())
	{
		file.seekg(0, std::ifstream::beg);

		// read node and element count
		std::string line;
		std::getline(file, line, '\n');
		int node_cnt = std::stoi(line);

		std::getline(file, line, '\n');
		int element_cnt = std::stoi(line);

		unsigned int lines_read = 0;
		while (!file.eof())
		{
			std::string line;
			std::getline(file, line, '\n');
			std::stringstream ss(line);
			std::string linestart = line.substr(0, 2);

			if (std::strcmp("//", linestart.c_str()) != 0 && line.length() > 0)
			{
				if (lines_read < node_cnt)
				{
					std::string x, y, z;
					ss >> x >> y >> z;

					nodes.push_back(std::make_tuple<float, float, float>(std::stof(x), std::stof(y), std::stof(z)));
				}
				else
				{
					std::string type;
					ss >> type;

					if (std::stoi(type) == 2)
					{
						std::string idx0, idx1, idx2, idx3;

						ss >> idx0 >> idx1 >> idx2 >> idx3;

						floor_elements.push_back(std::make_tuple<int, int, int, int>(
							std::stoi(idx0), std::stoi(idx1), std::stoi(idx2), std::stoi(idx3))
						);
					}
					else if (std::stoi(type) == 0)
					{
						std::string idx0, idx1;

						ss >> idx0 >> idx1;

						beam_elements.push_back(std::make_tuple<int, int>(
							std::stoi(idx0), std::stoi(idx1))
						);
					}
					else if (std::stoi(type) == 1)
					{
						std::string idx0, idx1;

						ss >> idx0 >> idx1;

						diagonal_elements.push_back(std::make_tuple<int, int>(
							std::stoi(idx0), std::stoi(idx1))
						);
					}
				}

				lines_read++;
			}
		}
	}
}

bool ArchVisMSMDataSource::load(std::string const& shader_filename,
	std::string const& partsList_filename,
	std::string const& nodesElement_filename)
{
	std::cout << "loading data" << std::endl;

	/** Nodes of the MSM */
	std::vector<Node> nodes;

	/** Floor elements of the MSM */
	std::vector<FloorElement> floor_elements;

	/** Beam elements of the MSM */
	std::vector<BeamElement> beam_elements;

	/** Diagonal elements of the MSM */
	std::vector<DiagonalElement> diagonal_elements;

	// Load node and element data
	parseNodeElementTable(nodesElement_filename, nodes, floor_elements, beam_elements, diagonal_elements);


	ShaderPrgmDataAccessor				shader_prgm_data;
	MeshDataAccessor					mesh_data;
	DrawCommandDataAccessor				draw_command_data;
	ObjectShaderParamsDataAccessor		mesh_shader_params;
	MaterialShaderParamsDataAccessor	mtl_shader_params;

	shader_prgm_data.char_cnt = shader_filename.length();
	shader_prgm_data.raw_string = new char[shader_prgm_data.char_cnt];
	std::strcpy(shader_prgm_data.raw_string, shader_filename.c_str());
	
	// parse parts list
	std::vector<std::string> parts_meshes_paths = parsePartsList(partsList_filename);

	// Create vector of glTF models
	std::vector<tinygltf::Model> models;
	tinygltf::TinyGLTF loader;
	std::string err;

	for (auto path : parts_meshes_paths)
	{
		models.push_back(tinygltf::Model());

		bool ret = loader.LoadASCIIFromFile(&models.back(), &err, path);
		if (!err.empty()) {
			printf("Err: %s\n", err.c_str());
		}

		if (!ret) {
			printf("Failed to parse glTF\n");
		}
	}

	// set vertex attribute layout
	std::vector<std::string> attribute_names;
	mesh_data.vertex_descriptor.attribute_cnt = models.front().meshes.front().primitives.front().attributes.size();
	mesh_data.vertex_descriptor.attributes = new MeshDataAccessor::VertexLayoutData::Attribute[mesh_data.vertex_descriptor.attribute_cnt];
	mesh_data.vertex_descriptor.stride = 0; //we currently cannot handle a different stride per attribute buffer, assuming data to be tightly packed
	size_t attribute_counter = 0;
	for (auto& attribute : models.front().meshes.front().primitives.front().attributes)
	{
		std::cout << "Attribute " << attribute.first << std::endl;

		attribute_names.push_back(attribute.first);

		std::cout << models.front().accessors[attribute.second].type << std::endl;
		std::cout << models.front().accessors[attribute.second].componentType << std::endl;
		std::cout << models.front().accessors[attribute.second].count << std::endl;
		std::cout << models.front().accessors[attribute.second].byteOffset << std::endl;
		std::cout << models.front().accessors[attribute.second].normalized << std::endl;

		//auto& bufferView = models.front().bufferViews[models.front().accessors[attribute.second].bufferView];

		mesh_data.vertex_descriptor.attributes[attribute_counter].type = models.front().accessors[attribute.second].componentType;
		mesh_data.vertex_descriptor.attributes[attribute_counter].size = models.front().accessors[attribute.second].type;
		mesh_data.vertex_descriptor.attributes[attribute_counter].normalized = models.front().accessors[attribute.second].normalized;
		// we use a seperated VBO per vertex attribute, therefore offset should always be zero...
		mesh_data.vertex_descriptor.attributes[attribute_counter].offset = 0;

		++attribute_counter;
	}

	// Create mesh data that holds all models
	// For now, assume glTF will supply non-interleaved data (this seems to be mostly the case,..how do I even identify interleaved data?)
	// Also assume that all glTF files contain a single mesh and all meshes use the same vertex format (i.e. go into a single render batch)
	mesh_data.index_data.byte_size = 0;
	mesh_data.vertex_data.byte_size = 0;
	mesh_data.vertex_data.buffer_cnt = mesh_data.vertex_descriptor.attribute_cnt;

	std::vector<uint32_t> first_indices; // index of the first index of the different meshes stored in the buffer
	std::vector<uint32_t> indices_cnt;
	std::vector<uint32_t> base_vertices; // base vertices of the different meshes stored in the buffer
	first_indices.push_back(0);
	base_vertices.push_back(0);

	// Sum up required storage for index and vertex data
	for (auto& model : models)
	{
		auto index_buffer_accessor = model.accessors[model.meshes.front().primitives.front().indices];
		auto index_bufferView = model.bufferViews[index_buffer_accessor.bufferView];
		mesh_data.index_data.index_type = index_buffer_accessor.componentType;
		mesh_data.index_data.byte_size += index_bufferView.byteLength;

		for (auto& attrib : model.meshes.front().primitives.front().attributes)
		{
			auto& accessor = model.accessors[attrib.second];
			auto& bufferView = model.bufferViews[accessor.bufferView];

			mesh_data.vertex_data.byte_size += bufferView.byteLength;
		}

		// log first index and base vertex for each model
		uint32_t base_vertex = base_vertices.back() + model.accessors[model.meshes.front().primitives.front().attributes.at(attribute_names.front())].count;
		base_vertices.push_back(base_vertex);

		uint32_t first_index = first_indices.back() + index_buffer_accessor.count;
		first_indices.push_back(first_index);

		indices_cnt.push_back(index_buffer_accessor.count);
	}
	// need additional storage for storing byte offsets of individual buffers (4byte per buffer)
	mesh_data.vertex_data.byte_size += 4 * mesh_data.vertex_data.buffer_cnt;

	// Allocate shared buffer for vertex data and copy data from gltf to buffer, start after offset values
	mesh_data.vertex_data.raw_data = new uint8_t[mesh_data.vertex_data.byte_size];
	uint32_t* uint32_view = reinterpret_cast<uint32_t*>(mesh_data.vertex_data.raw_data);
	uint32_t bytes_copied = 4 * mesh_data.vertex_data.buffer_cnt;

	for (int i = 0; i < mesh_data.vertex_data.buffer_cnt; ++i)
	{
		uint32_view[i] = bytes_copied;

		for (auto& model : models)
		{
			auto& accessor = model.accessors[model.meshes.front().primitives.front().attributes.at(attribute_names[i])];
			auto& bufferView = model.bufferViews[accessor.bufferView];

			auto tgt = mesh_data.vertex_data.raw_data + bytes_copied;
			auto src = model.buffers[bufferView.buffer].data.data() + accessor.byteOffset + bufferView.byteOffset;
			auto size = bufferView.byteLength;

			std::memcpy(tgt, src, size);

			bytes_copied += size;
		}
	}

	// Allocate shared buffer for index data and copy data from gltf to buffer
	mesh_data.index_data.raw_data = new uint8_t[mesh_data.index_data.byte_size];

	bytes_copied = 0;
	for (auto& model : models)
	{
		auto& accessor = model.accessors[model.meshes.front().primitives.front().indices];
		auto& bufferView = model.bufferViews[accessor.bufferView];

		auto tgt = mesh_data.index_data.raw_data + bytes_copied;
		auto src = model.buffers[bufferView.buffer].data.data() + accessor.byteOffset + bufferView.byteOffset;
		auto size = bufferView.byteLength;

		std::memcpy(tgt, src, size);

		bytes_copied += size;
	}

	assert(bytes_copied == mesh_data.index_data.byte_size);

	//std::cout << "==========================" << std::endl;
	//float* float_view = reinterpret_cast<float*>(mesh_data.vertex_data.raw_data + 16);
	//for (int i = 0; i < base_vertices.back() * 8; ++i)
	//{
	//	std::cout << float_view[i] << std::endl;
	//}
	//std::cout << "==========================" << std::endl;
	//
	//std::cout << "==========================" << std::endl;
	//uint16_t* uint_view = reinterpret_cast<uint16_t*>(mesh_data.index_data.raw_data);
	//for (int i = 0; i < first_indices.back(); ++i)
	//{
	//	std::cout << uint_view[i] << std::endl;
	//}
	//std::cout << "==========================" << std::endl;


	// TODO Build initial transform matrices from node/element data

	std::mt19937 generator(4215);
	std::uniform_real_distribution<float> distr(0.05f, 0.1f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	draw_command_data.draw_cnt = floor_elements.size() + beam_elements.size() + diagonal_elements.size();
	draw_command_data.data = new DrawCommandDataAccessor::DrawElementsCommand[draw_command_data.draw_cnt];

	mesh_shader_params.byte_size = 16 * 4 * draw_command_data.draw_cnt;
	mesh_shader_params.raw_data = new uint8_t[mesh_shader_params.byte_size];

	int counter = 0;
	for (auto& element : floor_elements)
	{
		draw_command_data.data[counter].cnt = indices_cnt[2];
		draw_command_data.data[counter].instance_cnt = 1;
		draw_command_data.data[counter].first_idx = first_indices[2];
		draw_command_data.data[counter].base_vertex = base_vertices[2];
		draw_command_data.data[counter].base_instance = 0;

		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;
		GLfloat scale = distr(generator);
		scale = 1.0f;
		object_transform.SetAt(0, 0, scale);
		object_transform.SetAt(1, 1, scale);
		object_transform.SetAt(2, 2, scale);

		object_transform.SetAt(0, 3, std::get<0>(nodes[std::get<3>(element)]));
		object_transform.SetAt(1, 3, std::get<1>(nodes[std::get<3>(element)]) -1.0f);
		object_transform.SetAt(2, 3, std::get<2>(nodes[std::get<3>(element)]));

		std::memcpy(mesh_shader_params.raw_data + counter*(16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	for (auto& element : beam_elements)
	{
		draw_command_data.data[counter].cnt = indices_cnt[0];
		draw_command_data.data[counter].instance_cnt = 1;
		draw_command_data.data[counter].first_idx = first_indices[0];
		draw_command_data.data[counter].base_vertex = base_vertices[0];
		draw_command_data.data[counter].base_instance = 0;

		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;
		GLfloat scale = distr(generator);
		scale = 1.0f;
		object_transform.SetAt(0, 0, scale);
		object_transform.SetAt(1, 1, scale);
		object_transform.SetAt(2, 2, scale);

		object_transform.SetAt(0, 3, std::get<0>(nodes[std::get<0>(element)]));
		object_transform.SetAt(1, 3, std::get<1>(nodes[std::get<0>(element)]) - 1.0f);
		object_transform.SetAt(2, 3, std::get<2>(nodes[std::get<0>(element)]));

		std::memcpy(mesh_shader_params.raw_data + counter*(16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	for (auto& element : diagonal_elements)
	{
		draw_command_data.data[counter].cnt = indices_cnt[1];
		draw_command_data.data[counter].instance_cnt = 1;
		draw_command_data.data[counter].first_idx = first_indices[1];
		draw_command_data.data[counter].base_vertex = base_vertices[1];
		draw_command_data.data[counter].base_instance = 0;

		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;

		// compute element rotation
		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_rotation;
		vislib::math::Vector<float, 3> diag_vector = vislib::math::Vector<float, 3>(std::get<0>(nodes[std::get<1>(element)]) - std::get<0>(nodes[std::get<0>(element)]),
			std::get<1>(nodes[std::get<1>(element)]) - std::get<1>(nodes[std::get<0>(element)]),
			std::get<2>(nodes[std::get<1>(element)]) - std::get<2>(nodes[std::get<0>(element)]));
		diag_vector.Normalise();
		vislib::math::Vector<float, 3> up_vector(0.0f, 1.0f, 0.0f);
		vislib::math::Vector<float, 3> rot_vector = up_vector.Cross(diag_vector);
		rot_vector.Normalise();
		vislib::math::Quaternion<float> rotation( std::acos(up_vector.Dot(diag_vector)), rot_vector);
		object_rotation = rotation;

		// compute element offset
		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_translation;
		object_translation.SetAt(0, 3, std::get<0>(nodes[std::get<0>(element)]));
		object_translation.SetAt(1, 3, std::get<1>(nodes[std::get<0>(element)]) - 1.0f);
		object_translation.SetAt(2, 3, std::get<2>(nodes[std::get<0>(element)]));

		object_transform = object_translation * object_rotation;

		std::memcpy(mesh_shader_params.raw_data + counter*(16 * 4), object_transform.PeekComponents(), 16 * 4);

		counter++;
	}

	/*
	for (int i = 0; i < draw_command_data.draw_cnt; ++i)
	{
		draw_command_data.data[i].cnt = index_buffer_accessor.count;
		draw_command_data.data[i].instance_cnt = 1;
		draw_command_data.data[i].first_idx = 0;
		draw_command_data.data[i].base_vertex = 0;
		draw_command_data.data[i].base_instance = 0;


		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;
		GLfloat scale = distr(generator);
		scale = 0.1f;
		object_transform.SetAt(0, 0, scale);
		object_transform.SetAt(1, 1, scale);
		object_transform.SetAt(2, 2, scale);

		object_transform.SetAt(0, 3, loc_distr(generator));
		object_transform.SetAt(1, 3, loc_distr(generator));
		object_transform.SetAt(2, 3, loc_distr(generator));

		std::memcpy(mesh_shader_params.raw_data + i*(16 * 4), object_transform.PeekComponents(), 16 * 4);
	}
	*/

	mtl_shader_params.elements_cnt = 0;
	
	m_render_batches.addBatch(
		shader_prgm_data,
		mesh_data,
		draw_command_data,
		mesh_shader_params,
		mtl_shader_params);

	/*
	mesh_data.vertex_data.byte_size = 3 * 6 * 4;
	mesh_data.vertex_data.raw_data = new uint8_t[mesh_data.vertex_data.byte_size]; // 3 triangles * 6 float entries * bytesize
	float* float_view = reinterpret_cast<float*>(mesh_data.vertex_data.raw_data);
	float_view[0] = -0.5f;
	float_view[1] = 0.0f;
	float_view[2] = 0.0f;
	float_view[3] = 0.0f;
	float_view[4] = 0.0f;
	float_view[5] = 1.0f;

	float_view[6] = 0.5f;
	float_view[7] = 0.0f;
	float_view[8] = 0.0f;
	float_view[9] = 0.0f;
	float_view[10] = 0.0f;
	float_view[11] = 1.0f;

	float_view[12] = 0.0f;
	float_view[13] = 1.0f;
	float_view[14] = 0.0f;
	float_view[15] = 0.0f;
	float_view[16] = 0.0f;
	float_view[17] = 1.0f;

	mesh_data.index_data.index_type = GL_UNSIGNED_INT;
	mesh_data.index_data.byte_size = 3 * 4;
	mesh_data.index_data.raw_data = new uint8_t[3 * 4];
	uint32_t* uint_view = reinterpret_cast<uint32_t*>(mesh_data.index_data.raw_data);
	uint_view[0] = 0;
	uint_view[1] = 1;
	uint_view[2] = 2;

	mesh_data.vertex_descriptor.stride = 24;
	mesh_data.vertex_descriptor.attribute_cnt = 2;
	mesh_data.vertex_descriptor.attributes = new MeshDataAccessor::VertexLayoutData::Attribute[mesh_data.vertex_descriptor.attribute_cnt];
	mesh_data.vertex_descriptor.attributes[0].type = GL_FLOAT;
	mesh_data.vertex_descriptor.attributes[0].size = 3;
	mesh_data.vertex_descriptor.attributes[0].normalized = GL_FALSE;
	mesh_data.vertex_descriptor.attributes[0].offset = 0;
	mesh_data.vertex_descriptor.attributes[1].type = GL_FLOAT;
	mesh_data.vertex_descriptor.attributes[1].size = 3;
	mesh_data.vertex_descriptor.attributes[1].normalized = GL_FALSE;
	mesh_data.vertex_descriptor.attributes[1].offset = 12;
	

	std::mt19937 generator(4215);
	std::uniform_real_distribution<float> distr(0.05f, 0.1f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	draw_command_data.draw_cnt = 1000000;
	draw_command_data.data = new DrawCommandDataAccessor::DrawElementsCommand[draw_command_data.draw_cnt];

	mesh_shader_params.byte_size = 16 * 4 * draw_command_data.draw_cnt;
	mesh_shader_params.raw_data = new uint8_t[mesh_shader_params.byte_size];

	for (int i = 0; i < draw_command_data.draw_cnt; ++i)
	{
		draw_command_data.data[i].cnt = 3;
		draw_command_data.data[i].instance_cnt = 1;
		draw_command_data.data[i].first_idx = 0;
		draw_command_data.data[i].base_vertex = 0;
		draw_command_data.data[i].base_instance = 0;


		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;
		GLfloat scale = distr(generator);
		object_transform.SetAt(0, 0, scale);
		object_transform.SetAt(1, 1, scale);
		object_transform.SetAt(2, 2, scale);

		object_transform.SetAt(0, 3, loc_distr(generator));
		object_transform.SetAt(1, 3, loc_distr(generator));
		object_transform.SetAt(2, 3, loc_distr(generator));

		std::memcpy(mesh_shader_params.raw_data + i*(16 * 4), object_transform.PeekComponents(), 16 * 4);
	}

	mtl_shader_params.elements_cnt = 0;

	m_render_batches.addBatch(
		shader_prgm_data,
		mesh_data,
		draw_command_data,
		mesh_shader_params,
		mtl_shader_params);
	*/

	
	return true;
}