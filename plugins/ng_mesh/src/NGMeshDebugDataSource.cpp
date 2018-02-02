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

#include "stdafx.h"
#include "NGMeshDebugDataSource.h"

using namespace megamol;
using namespace megamol::ngmesh;

NGMeshDebugDataSource::NGMeshDebugDataSource() :
	m_shaderFilename_slot("shader filename", "The name of to the shader file to load"),
	m_geometryFilename_slot("mesh filename", "The path to the mesh file to load")
{
	this->m_shaderFilename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_shaderFilename_slot);

	this->m_geometryFilename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_geometryFilename_slot);
}

NGMeshDebugDataSource::~NGMeshDebugDataSource()
{
}

bool NGMeshDebugDataSource::getDataCallback(core::Call& caller)
{
	CallNGMeshRenderBatches* render_batches_call = dynamic_cast<CallNGMeshRenderBatches*>(&caller);
	if (render_batches_call == NULL)
		return false;
	if (this->m_geometryFilename_slot.IsDirty() || this->m_shaderFilename_slot.IsDirty())
	{
		this->m_geometryFilename_slot.ResetDirty();
		this->m_shaderFilename_slot.ResetDirty();

		// Clear render batches TODO: add explicit clear function?
		CallNGMeshRenderBatches::RenderBatchesData empty_render_batches;
		m_render_batches = empty_render_batches;

		m_bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

		auto vislib_shader_filename = m_shaderFilename_slot.Param<core::param::FilePathParam>()->Value();
		std::string shdr_filename(vislib_shader_filename.PeekBuffer());

		auto vislib_geometry_filename = m_geometryFilename_slot.Param<core::param::FilePathParam>()->Value();
		std::string geom_filename(vislib_geometry_filename.PeekBuffer());

		load(shdr_filename, geom_filename);
	}

	render_batches_call->setRenderBatches(&m_render_batches);

	return true;
}

bool NGMeshDebugDataSource::load(std::string const& shader_filename, std::string const& geometry_filename)
{
	ShaderPrgmDataAccessor				shader_prgm_data;
	MeshDataAccessor					mesh_data;
	DrawCommandDataAccessor				draw_command_data;
	ObjectShaderParamsDataAccessor		obj_shader_params;
	MaterialShaderParamsDataAccessor	mtl_shader_params;

	shader_prgm_data.char_cnt = shader_filename.length();
	shader_prgm_data.raw_string = new char[shader_prgm_data.char_cnt];
	std::strcpy(shader_prgm_data.raw_string, shader_filename.c_str());

	mesh_data.vertex_data.buffer_cnt = 2;
	mesh_data.vertex_data.byte_size = 3 * 6 * 4 + (4 * mesh_data.vertex_data.buffer_cnt);
	mesh_data.vertex_data.raw_data = new uint8_t[mesh_data.vertex_data.byte_size]; // 3 triangles * 6 float entries * bytesize
	uint32_t* uint32_view = reinterpret_cast<uint32_t*>(mesh_data.vertex_data.raw_data);

	uint32_view[0] = 8;
	uint32_view[1] = 8 + 9*4;

	float* float_view = reinterpret_cast<float*>(mesh_data.vertex_data.raw_data + 8);
	// position attribute
	float_view[0] = -0.5f;
	float_view[1] = 0.0f;
	float_view[2] = 0.0f;
	float_view[3] = 0.5f;
	float_view[4] = 0.0f;
	float_view[5] = 0.0f;
	float_view[6] = 0.0f;
	float_view[7] = 1.0f;
	float_view[8] = 0.0f;

	// normal attribute
	float_view[9] = 0.0f;
	float_view[10] = 0.0f;
	float_view[11] = 1.0f;
	float_view[12] = 0.0f;
	float_view[13] = 0.0f;
	float_view[14] = 1.0f;
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

	//mesh_data.vertex_descriptor.stride = 24;
	mesh_data.vertex_descriptor.stride = 0;
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
	std::uniform_real_distribution<float> distr(0.01f, 0.03f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	draw_command_data.draw_cnt = 1000000;
	draw_command_data.data = new DrawCommandDataAccessor::DrawElementsCommand[draw_command_data.draw_cnt];

	obj_shader_params.byte_size = 16 * 4 * draw_command_data.draw_cnt;
	obj_shader_params.raw_data = new uint8_t[obj_shader_params.byte_size];

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

		std::memcpy(obj_shader_params.raw_data + i*(16 * 4), object_transform.PeekComponents(), 16 * 4);
	}

	mtl_shader_params.elements_cnt = 0;


	m_render_batches.addBatch(
		shader_prgm_data,
		mesh_data,
		draw_command_data,
		obj_shader_params,
		mtl_shader_params);
	
	return true;
}