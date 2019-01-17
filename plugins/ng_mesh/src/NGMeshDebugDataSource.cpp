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

#include "ng_mesh/NGMeshRenderBatchBakery.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

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

bool NGMeshDebugDataSource::load(std::string const& shader_btf_namespace, std::string const& geometry_filename)
{
	ShaderPrgmDataAccessor				shader_prgm_data;
	DrawCommandsDataAccessor				draw_command_data;
	ObjectShaderParamsDataAccessor		obj_shader_params;
	MaterialShaderParamsDataAccessor	mtl_shader_params;

	shader_prgm_data.char_cnt = shader_btf_namespace.length();
	shader_prgm_data.raw_string = new char[shader_prgm_data.char_cnt];
	std::strcpy(shader_prgm_data.raw_string, shader_btf_namespace.c_str());

	///////////
	// Deprecated MeshDataAccessor initialization
	///////////
	//MeshDataAccessor mesh_data;
	//mesh_data.usage = GL_STATIC_DRAW;
	//mesh_data.primitive_type = GL_TRIANGLES;
	//
	//mesh_data.vertex_data.buffer_cnt = 2;
	//std::vector<MeshDataAccessor::VertexData::Buffer> buffers(mesh_data.vertex_data.buffer_cnt);
	//mesh_data.vertex_data.buffers = buffers.data();
	//
	//std::vector<uint8_t> vertex_data(3 * 6 * 4); // 3 vertices * 6 float entries * bytesize
	//mesh_data.vertex_data.buffers[0].raw_data = vertex_data.data();
	//mesh_data.vertex_data.buffers[0].byte_size = 3 * 3 * 4; // 3 vertices * 3 float entries for position * bytesize
	//mesh_data.vertex_data.buffers[1].raw_data = vertex_data.data() + (3 * 3 * 4);
	//mesh_data.vertex_data.buffers[1].byte_size = 3 * 3 * 4; // 3 vertices * 3 float entries for normal * bytesize
	//
	//float* float_view = reinterpret_cast<float*>(mesh_data.vertex_data.buffers[0].raw_data);
	//// position attribute
	//float_view[0] = -0.5f;
	//float_view[1] = 0.0f;
	//float_view[2] = 0.0f;
	//float_view[3] = 0.5f;
	//float_view[4] = 0.0f;
	//float_view[5] = 0.0f;
	//float_view[6] = 0.0f;
	//float_view[7] = 1.0f;
	//float_view[8] = 0.0f;
	//
	//// normal attribute
	//float_view[9] = 0.0f;
	//float_view[10] = 0.0f;
	//float_view[11] = 1.0f;
	//float_view[12] = 0.0f;
	//float_view[13] = 0.0f;
	//float_view[14] = 1.0f;
	//float_view[15] = 0.0f;
	//float_view[16] = 0.0f;
	//float_view[17] = 1.0f;
	//
	//mesh_data.index_data.index_type = GL_UNSIGNED_INT;
	//mesh_data.index_data.byte_size = 3 * 4;
	//mesh_data.index_data.raw_data = new uint8_t[3 * 4];
	//uint32_t* uint_view = reinterpret_cast<uint32_t*>(mesh_data.index_data.raw_data);
	//uint_view[0] = 0;
	//uint_view[1] = 1;
	//uint_view[2] = 2;
	//
	////mesh_data.vertex_descriptor.stride = 24;
	//mesh_data.vertex_descriptor.stride = 0;
	//mesh_data.vertex_descriptor.attribute_cnt = 2;
	//mesh_data.vertex_descriptor.attributes = new MeshDataAccessor::VertexLayoutData::Attribute[mesh_data.vertex_descriptor.attribute_cnt];
	//mesh_data.vertex_descriptor.attributes[0].type = GL_FLOAT;
	//mesh_data.vertex_descriptor.attributes[0].size = 3;
	//mesh_data.vertex_descriptor.attributes[0].normalized = GL_FALSE;
	//mesh_data.vertex_descriptor.attributes[0].offset = 0;
	//mesh_data.vertex_descriptor.attributes[1].type = GL_FLOAT;
	//mesh_data.vertex_descriptor.attributes[1].size = 3;
	//mesh_data.vertex_descriptor.attributes[1].normalized = GL_FALSE;
	//mesh_data.vertex_descriptor.attributes[1].offset = 0;

	
	// Create std-container for holding vertex data
    std::vector<std::vector<float>> vbs = {{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f},// normal data buffer
        {-0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}}; // position data buffer
	// Create std-container holding vertex attribute descriptions
	std::vector<VertexLayout::Attribute> attribs = {
		VertexLayout::Attribute(3,GL_FLOAT,GL_FALSE,0),
		VertexLayout::Attribute(3,GL_FLOAT,GL_FALSE,0) };

	// Create std-container holding index data
	std::vector<uint32_t> indices = { 0,1,2 };

    /*
	// Build vertex buffer accessor data for vertex buffers
	auto vb_accs = buildVertexBufferAccessors(vbs);
	// Build mesh accessor from available data + additional info on mesh
	MeshDataAccessor mesh_acc = buildMeshDataAccessor(
		vb_accs,
		attribs,
		0,
		indices,
		GL_UNSIGNED_INT,
		GL_STATIC_DRAW,
		GL_TRIANGLES);
	/*
	std::mt19937 generator(4215);
	std::uniform_real_distribution<float> distr(0.01f, 0.03f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	std::vector<DrawCommandDataAccessor::DrawElementsCommand> draw_commands(100);
	draw_command_data.draw_cnt = draw_commands.size();
	draw_command_data.data = draw_commands.data();

	std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> object_transforms(100);
	obj_shader_params.byte_size = object_transforms.size() * sizeof(vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>);
	obj_shader_params.raw_data = reinterpret_cast<uint8_t*>( object_transforms.data() );

	for (int i = 0; i < draw_command_data.draw_cnt; ++i)
	{
		draw_command_data.data[i].cnt = 3;
		draw_command_data.data[i].instance_cnt = 1;
		draw_command_data.data[i].first_idx = 0;
		draw_command_data.data[i].base_vertex = 0;
		draw_command_data.data[i].base_instance = 0;

		GLfloat scale = distr(generator);
		scale = 1.0f;
		object_transforms[i].SetAt(0, 0, scale);
		object_transforms[i].SetAt(1, 1, scale);
		object_transforms[i].SetAt(2, 2, scale);

		object_transforms[i].SetAt(0, 3, loc_distr(generator));
		object_transforms[i].SetAt(1, 3, loc_distr(generator));
		object_transforms[i].SetAt(2, 3, loc_distr(generator));
	}

	mtl_shader_params.elements_cnt = 0;
	*/

	
	auto mesh_data = loadGLTF(geometry_filename);

	auto vb_accs = buildVertexBufferAccessors(*std::get<1>(mesh_data));
    auto mesh_acc = buildMeshDataAccessor(vb_accs, *std::get<0>(mesh_data), 0, *std::get<2>(mesh_data), GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);

    draw_command_data.draw_cnt = std::get<3>(mesh_data)->size();
    draw_command_data.data = std::get<3>(mesh_data)->data();

	obj_shader_params.byte_size = std::get<4>(mesh_data)->size();
    obj_shader_params.raw_data = std::get<4>(mesh_data)->data();

    //vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transforms;
	//
    //float scale = 0.01f;
    //object_transforms.SetAt(0, 0, scale);
    //object_transforms.SetAt(1, 1, scale);
    //object_transforms.SetAt(2, 2, scale);
	//
    //object_transforms.SetAt(0, 3, 0.0f);
    //object_transforms.SetAt(1, 3, 0.0f);
    //object_transforms.SetAt(2, 3, 0.0f);

	mtl_shader_params.elements_cnt = 0;

	m_render_batches.addBatch(
		shader_prgm_data,
		mesh_acc,
		draw_command_data,
		obj_shader_params,
		mtl_shader_params);
	
	return true;
}

std::tuple<std::shared_ptr<std::vector<VertexLayout::Attribute>>,
    std::shared_ptr<std::vector<std::vector<uint8_t>>>,
    std::shared_ptr<std::vector<uint8_t>>,
    std::shared_ptr<std::vector<DrawCommandsDataAccessor::DrawElementsCommand>>,
    std::shared_ptr<std::vector<uint8_t>>>
	NGMeshDebugDataSource::loadGLTF(std::string const& geometry_filename) {
    // Create vector of glTF models
	tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string war;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &war, geometry_filename);
    if (!err.empty()) {
        vislib::sys::Log::DefaultLog.WriteError("Err: %s\n", err.c_str());
    }

    if (!ret) {
        vislib::sys::Log::DefaultLog.WriteError("Failed to parse glTF\n");
    }

	// Create std-container holding vertex attribute descriptions
    std::vector<std::string> attribute_names;
    std::shared_ptr<std::vector<VertexLayout::Attribute>> attribs = std::make_shared<std::vector<VertexLayout::Attribute>>();
    for (auto& attribute : model.meshes.front().primitives.front().attributes) {
        attribute_names.push_back(attribute.first);
        attribs->push_back(VertexLayout::Attribute(model.accessors[attribute.second].type,
            model.accessors[attribute.second].componentType,
            model.accessors[attribute.second].normalized,
            0)); // a seperated VBO is used per vertex attribute, therefore offset should always be zero...
    }

	// Create intermediate information storage for index data
    size_t index_buffer_byteSize = 0;
    std::vector<uint32_t> first_indices = {0}; // index of the first index of the different meshes stored in the buffer
    std::vector<uint32_t> indices_cnt;
    GLenum index_type = GL_UNSIGNED_INT;

    // Create intermediate information storage for vertex data
    std::vector<size_t> vertex_buffers_byteSize(attribs->size(), 0);
    std::vector<uint32_t> base_vertices = {0}; // base vertices of the different meshes stored in the buffer

	size_t mesh_node_cnt = 0;

	// Scan all nodes to sum up required storage for index and vertex data
	for (auto& node : model.nodes)
	{
        if (node.mesh == -1) continue;

        auto& indices_accessor = model.accessors[model.meshes[node.mesh].primitives.front().indices];
        auto& indices_bufferView = model.bufferViews[indices_accessor.bufferView];
        index_type = indices_accessor.componentType; // TODO: detect different index types and throw error?
        index_buffer_byteSize += indices_bufferView.byteLength;

        int attrib_counter = 0;
        for (auto& attrib : model.meshes[node.mesh].primitives.front().attributes) {
            auto& accessor = model.accessors[attrib.second];
            auto& bufferView = model.bufferViews[accessor.bufferView];

            vertex_buffers_byteSize[attrib_counter] += bufferView.byteLength;
            ++attrib_counter;
        }

        // log first index and base vertex for each node
        uint32_t base_vertex =
            base_vertices.back() +
            model.accessors[model.meshes[node.mesh].primitives.front().attributes.at(attribute_names.front())].count;
        base_vertices.push_back(base_vertex);

        uint32_t first_index = first_indices.back() + indices_accessor.count;
        first_indices.push_back(first_index);

        indices_cnt.push_back(indices_accessor.count);

		++mesh_node_cnt;
    }

	// Create intermediate data storage for index and vertex buffers (size are know after first scan of all models)

    // index data storage, index data from all models is gathered here (yes, this requires some copying)
    std::shared_ptr<std::vector<uint8_t>> index_data = std::make_shared<std::vector<uint8_t>>(index_buffer_byteSize); 

	// vertex data storage, vertex data from all models is gathered per attribute (yes, this requires some copying)
    std::shared_ptr<std::vector<std::vector<uint8_t>>> vbs = std::make_shared<std::vector<std::vector<uint8_t>>>();

    for (auto& size : vertex_buffers_byteSize)
		vbs->push_back(std::vector<uint8_t>(size));

	// Copy vertex data from gltf models to intermediate buffer
    for (int i = 0; i < vbs->size(); ++i) {
        uint32_t bytes_copied = 0;

        for (auto& node : model.nodes) {
            if (node.mesh == -1) continue;

            auto& accessor = model.accessors[model.meshes[node.mesh].primitives.front().attributes.at(attribute_names[i])];
            auto& bufferView = model.bufferViews[accessor.bufferView];

            auto tgt = (*vbs)[i].data() + bytes_copied;
            auto src = model.buffers[bufferView.buffer].data.data() + accessor.byteOffset + bufferView.byteOffset;
            auto size = bufferView.byteLength;

            std::memcpy(tgt, src, size);

            bytes_copied += size;
        }
    }

    // Copy index data from gltf models to intermediate buffer
    uint32_t bytes_copied = 0;
    for (auto& node : model.nodes) {
        if (node.mesh == -1) continue;

        auto& accessor = model.accessors[model.meshes[node.mesh].primitives.front().indices];
        auto& bufferView = model.bufferViews[accessor.bufferView];

        auto tgt = index_data->data() + bytes_copied;
        auto src = model.buffers[bufferView.buffer].data.data() + accessor.byteOffset + bufferView.byteOffset;
        auto size = bufferView.byteLength;

        std::memcpy(tgt, src, size);

        bytes_copied += size;
    }

    std::shared_ptr<std::vector<DrawCommandsDataAccessor::DrawElementsCommand>> draw_commands =
        std::make_shared<std::vector<DrawCommandsDataAccessor::DrawElementsCommand>>(mesh_node_cnt);

	std::shared_ptr<std::vector<uint8_t>> per_obj_shader_params = std::make_shared<std::vector<uint8_t>>(
        mesh_node_cnt * sizeof(vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>));

    for (int i = 0; i < mesh_node_cnt; ++i) {

        (*draw_commands)[i].cnt = indices_cnt[i];
        (*draw_commands)[i].instance_cnt = 1;
        (*draw_commands)[i].first_idx = first_indices[i];
        (*draw_commands)[i].base_vertex = base_vertices[i];
        (*draw_commands)[i].base_instance = 0;

		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transforms;
		float scale = 0.01f;
        object_transforms.SetAt(0, 0, scale);
        object_transforms.SetAt(1, 1, scale);
        object_transforms.SetAt(2, 2, scale);

		object_transforms.SetAt(0, 3, 0.0f);
        object_transforms.SetAt(1, 3, 0.0f);
        object_transforms.SetAt(2, 3, 1.0f);

        std::memcpy(
            per_obj_shader_params->data() + i * sizeof(vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>),
            object_transforms.PeekComponents(), sizeof(vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>));
    }

	vislib::sys::Log::DefaultLog.WriteInfo("Mesh node count: %u", mesh_node_cnt);
	
	return {
        attribs,
        vbs,
		index_data,
		draw_commands,
		per_obj_shader_params
    };

    //auto& indices_accessor = model.accessors[model.meshes.front().primitives.back().indices];
    //auto& indices_bufferView = model.bufferViews[indices_accessor.bufferView];
    //auto& indices_buffer = model.buffers[indices_bufferView.buffer];
    //std::shared_ptr<std::vector<uint8_t>> indices = std::make_shared<std::vector<uint8_t>>(
    //    indices_buffer.data.begin() + indices_bufferView.byteOffset + indices_accessor.byteOffset,
    //    indices_buffer.data.begin() + indices_bufferView.byteOffset + indices_accessor.byteOffset +
    //        (indices_accessor.count * indices_accessor.ByteStride(indices_bufferView)));
	//
    //auto stride = indices_accessor.ByteStride(indices_bufferView);
    //auto index_type = indices_accessor.componentType;
	//
    //for (auto& attribute : model.meshes.front().primitives.front().attributes) {
    //    //attribs->push_back(VertexLayout::Attribute(model.accessors[attribute.second].type,
    //    //    model.accessors[attribute.second].componentType, model.accessors[attribute.second].normalized,
    //    //    0)); // a seperated VBO is used per vertex attribute, therefore offset should always be zero...
	//
    //    auto& vertexAttrib_accessor = model.accessors[attribute.second];
    //    auto& vertexAttrib_bufferView = model.bufferViews[vertexAttrib_accessor.bufferView];
    //    auto& vertexAttrib_buffer = model.buffers[vertexAttrib_bufferView.buffer];
    //    vbs->push_back(std::vector<uint8_t>(vertexAttrib_buffer.data.begin() + vertexAttrib_bufferView.byteOffset +
    //                                            vertexAttrib_accessor.byteOffset,
    //        vertexAttrib_buffer.data.begin() + vertexAttrib_bufferView.byteOffset +
    //            vertexAttrib_accessor.byteOffset +
    //            (vertexAttrib_accessor.count * vertexAttrib_accessor.ByteStride(vertexAttrib_bufferView))));
    //}
	//

    // Build vertex buffer accessor data for vertex buffers
    //auto vb_accs = buildVertexBufferAccessors(vbs);
    
	//return buildMeshDataAccessor(vb_accs, attribs, 0, indices, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);
}