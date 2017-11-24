/*
* AbstractNGMeshDataSource.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include <iostream>
#include <random>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "stdafx.h"
#include "ArchVisMSMDataSource.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "external/tiny_gltf.h"

using namespace megamol;
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

bool ArchVisMSMDataSource::getDataCallback(core::Call& caller)
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

		auto vislib_shader_filename = m_shaderFilename_slot.Param<core::param::FilePathParam>()->Value();
		std::string shdr_filename(vislib_shader_filename.PeekBuffer());

		auto vislib_partsList_filename = m_partsList_slot.Param<core::param::FilePathParam>()->Value();
		std::string partsList_filename(vislib_partsList_filename.PeekBuffer());

		load(shdr_filename, partsList_filename);
	}

	// TODO handle IP slot is dirty

	render_batches_call->setRenderBatches(&m_render_batches);

	return true;
}

bool ArchVisMSMDataSource::load(std::string const& shader_filename, std::string const& partsList_filename)
{
	std::cout << "loading data" << std::endl;	

	ShaderPrgmDataAccessor				shader_prgm_data;
	MeshDataAccessor					mesh_data;
	DrawCommandDataAccessor				draw_command_data;
	ObjectShaderParamsDataAccessor		mesh_shader_params;
	MaterialShaderParamsDataAccessor	mtl_shader_params;

	shader_prgm_data.char_cnt = shader_filename.length();
	shader_prgm_data.raw_string = new char[shader_prgm_data.char_cnt];
	std::strcpy(shader_prgm_data.raw_string, shader_filename.c_str());
	
	// TODO parse parts list

	// TODO create vector of glTF models

	// TODO create mesh data that holds all models

	// TODO log first index and base vertex for all individual models

	// Begin test gltf

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;

	bool ret = loader.LoadASCIIFromFile(&model, &err, partsList_filename);
	if (!err.empty()) {
		printf("Err: %s\n", err.c_str());
	}

	if (!ret) {
		printf("Failed to parse glTF\n");
	}


	// For now, assume glTF will supply non-interleaved data (this seems to be mostly the case,..how do I even identify interleaved data?)


	//mesh_data.vertex_descriptor.stride = model.bufferViews[model.accessors[model.meshes.front().primitives.front().attributes.begin()->second].bufferView].byteStride;
	mesh_data.vertex_descriptor.attribute_cnt = model.meshes.front().primitives.front().attributes.size();
	mesh_data.vertex_descriptor.attributes = new MeshDataAccessor::VertexLayoutData::Attribute[mesh_data.vertex_descriptor.attribute_cnt];
	size_t i = 0;
	for (auto& attribute : model.meshes.front().primitives.front().attributes)
	{
		std::cout << "Attribute "<< attribute.first << std::endl;

		std::cout << model.accessors[attribute.second].type << std::endl;
		std::cout << model.accessors[attribute.second].componentType << std::endl;
		std::cout << model.accessors[attribute.second].count << std::endl;
		std::cout << model.accessors[attribute.second].byteOffset << std::endl;
		std::cout << model.accessors[attribute.second].normalized << std::endl;

		auto& bufferView = model.bufferViews[model.accessors[attribute.second].bufferView];

		mesh_data.vertex_descriptor.attributes[i].type = model.accessors[attribute.second].componentType;
		mesh_data.vertex_descriptor.attributes[i].size = model.accessors[attribute.second].type;
		mesh_data.vertex_descriptor.attributes[i].normalized = model.accessors[attribute.second].normalized;
		// we use a seperated VBO per vertex attribute, therefore offset should always be zero...
		mesh_data.vertex_descriptor.attributes[i].offset = 0;

		++i;
	}

	// Create local datastorage for actual index and vertex buffer
	auto index_buffer_accessor = model.accessors[model.meshes.front().primitives.front().indices];
	auto index_bufferView = model.bufferViews[index_buffer_accessor.bufferView];
	mesh_data.index_data.index_type = index_buffer_accessor.componentType;
	mesh_data.index_data.byte_size = index_bufferView.byteLength;
	mesh_data.index_data.raw_data = new uint8_t[mesh_data.index_data.byte_size];
	assert((index_bufferView.byteOffset + index_bufferView.byteLength) <= model.buffers[index_bufferView.buffer].data.size());
	std::memcpy(mesh_data.index_data.raw_data,
		model.buffers[index_bufferView.buffer].data.data() + index_bufferView.byteOffset,
		mesh_data.index_data.byte_size);

	//std::cout << "==========================" << std::endl;
	//uint16_t* uint_view = reinterpret_cast<uint16_t*>(mesh_data.index_data.raw_data);
	//for (int i = 0; i < index_buffer_accessor.count; ++i)
	//{
	//	std::cout << uint_view[i] << std::endl;
	//}
	//std::cout << "==========================" << std::endl;


	// sum up required memory for all attributes
	mesh_data.vertex_descriptor.stride = 0; //we currently cannot handle a different stride per attribute buffer, assuming data to be tightly packed
	mesh_data.vertex_data.byte_size = 0;
	mesh_data.vertex_data.buffer_cnt = mesh_data.vertex_descriptor.attribute_cnt;
	for (auto& attrib : model.meshes.front().primitives.front().attributes)
	{
		auto& accessor = model.accessors[attrib.second];
		auto& bufferView = model.bufferViews[accessor.bufferView];

		mesh_data.vertex_data.byte_size += bufferView.byteLength;
	}

	// need additional storage for storing byte offsets of individual buffers (4byte per buffer)
	mesh_data.vertex_data.byte_size += 4 * mesh_data.vertex_data.buffer_cnt;

	// allocate memory for vertex data
	mesh_data.vertex_data.raw_data = new uint8_t[mesh_data.vertex_data.byte_size];
	uint32_t* uint32_view = reinterpret_cast<uint32_t*>(mesh_data.vertex_data.raw_data);

	// get vertex count
	size_t vertex_cnt = model.accessors[model.meshes.front().primitives.front().attributes.begin()->second].count;
	//TODO check if count matches for all attributes (it really should or else wtf...)

	//TODO copy data from gltf to buffers, start after offset values
	size_t bytes_copied = 4 * mesh_data.vertex_data.buffer_cnt;

	//TODO sort attribute to match shader?

	size_t attrib_counter = 0;
	for (auto& attrib : model.meshes.front().primitives.front().attributes)
	{
		auto& accessor = model.accessors[attrib.second];
		auto& bufferView = model.bufferViews[accessor.bufferView];

		auto tgt = mesh_data.vertex_data.raw_data + bytes_copied;
		auto src = model.buffers[bufferView.buffer].data.data() + accessor.byteOffset + bufferView.byteOffset;
		auto size = bufferView.byteLength;

		std::memcpy(tgt, src, size);

		uint32_view[attrib_counter] = bytes_copied;
		std::cout << "Offset " << attrib_counter << ": " << bytes_copied<<std::endl;
		attrib_counter++;

		bytes_copied += size;
	}

	//auto vertex_buffer_accessor_0 = model.accessors[model.meshes.front().primitives.front().attributes.begin()->second];
	//auto vertex_bufferView = model.bufferViews[vertex_buffer_accessor_0.bufferView];
	//mesh_data.vertex_data.byte_size = vertex_bufferView.byteLength;
	//mesh_data.vertex_data.raw_data = new uint8_t[mesh_data.vertex_data.byte_size];
	//assert((vertex_bufferView.byteOffset+ mesh_data.vertex_data.byte_size) <= model.buffers[vertex_bufferView.buffer].data.size());
	//std::memcpy(mesh_data.vertex_data.raw_data,
	//	model.buffers[vertex_bufferView.buffer].data.data() + vertex_bufferView.byteOffset,
	//	mesh_data.vertex_data.byte_size);

	//std::cout << "==========================" << std::endl;
	//float* float_view = reinterpret_cast<float*>(mesh_data.vertex_data.raw_data);
	//for (int i = 0; i < vertex_buffer_accessor_0.count * 2; ++i)
	//{
	//	std::cout << float_view[i] << std::endl;
	//}
	//std::cout << "==========================" << std::endl;

	std::cout << "Index type: " << mesh_data.index_data.index_type << std::endl;
	std::cout << "Index byte size: " << mesh_data.index_data.byte_size << std::endl;

	std::cout << "Vertex buffer byte size: " << bytes_copied << std::endl;
	std::cout << "Vertex buffer byte stride: " << mesh_data.vertex_descriptor.stride << std::endl;

	// End test gltf

	std::mt19937 generator(4215);
	std::uniform_real_distribution<float> distr(0.05f, 0.1f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	draw_command_data.draw_cnt = 1000;
	draw_command_data.data = new DrawCommandDataAccessor::DrawElementsCommand[draw_command_data.draw_cnt];

	mesh_shader_params.byte_size = 16 * 4 * draw_command_data.draw_cnt;
	mesh_shader_params.raw_data = new uint8_t[mesh_shader_params.byte_size];

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