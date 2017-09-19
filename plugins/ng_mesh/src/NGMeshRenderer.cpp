/*
* NGMeshRenderer.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "NGMeshRenderer.h"

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"

using namespace megamol::core;
using namespace megamol::ngmesh;

NGMeshRenderer::NGMeshRenderer()
	: Renderer3DModule(), m_renderBatches_callerSlot("getData", "Connects the mesh renderer with a mesh data source")
{
	this->m_renderBatches_callerSlot.SetCompatibleCall<CallNGMeshRenderBatchesDescription>();
	this->MakeSlotAvailable(&this->m_renderBatches_callerSlot);
}

NGMeshRenderer::~NGMeshRenderer()
{

}

bool NGMeshRenderer::create()
{
	//TODO generate debug render batch that doesn't rely on data call

	CallNGMeshRenderBatches::RenderBatchesData::ShaderPrgmData			shader_prgm_data;
	CallNGMeshRenderBatches::RenderBatchesData::MeshData				mesh_data;
	CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData			draw_command_data;
	CallNGMeshRenderBatches::RenderBatchesData::MeshShaderParams		mesh_shader_params;
	CallNGMeshRenderBatches::RenderBatchesData::MaterialShaderParams	mtl_shader_params;

	shader_prgm_data.raw_string = "NGMeshDebug";
	shader_prgm_data.char_cnt = 12;

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

	mesh_data.vertex_descriptor = VertexLayout(24, { VertexLayout::Attribute(GL_FLOAT,3,GL_FALSE,0),VertexLayout::Attribute(GL_FLOAT,3,GL_FALSE,12) });

	draw_command_data.draw_cnt = 1;
	draw_command_data.data = new CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData::DrawElementsCommand;
	draw_command_data.data->vertex_cnt = 3;
	draw_command_data.data->instance_cnt = 1;
	draw_command_data.data->first_idx = 0;
	draw_command_data.data->base_vertex = 0;
	draw_command_data.data->base_instance = 0;

	addRenderBatch(shader_prgm_data, mesh_data, draw_command_data, mesh_shader_params, mtl_shader_params);
	

	return true;
}

void NGMeshRenderer::release()
{

}

bool NGMeshRenderer::GetCapabilities(megamol::core::Call& call)
{
	//TODO
	return true;
}

bool NGMeshRenderer::GetExtents(megamol::core::Call& call)
{
	//TODO
	return true;
}

CallNGMeshRenderBatches* NGMeshRenderer::getData()
{
	CallNGMeshRenderBatches* rb_call = this->m_renderBatches_callerSlot.CallAs<CallNGMeshRenderBatches>();

	if (rb_call != NULL)
	{
		return rb_call;
	}
	else
	{
		return nullptr;
	}
}

void NGMeshRenderer::addRenderBatch(
	CallNGMeshRenderBatches::RenderBatchesData::ShaderPrgmData			shader_prgm_data,
	CallNGMeshRenderBatches::RenderBatchesData::MeshData				mesh_data,
	CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData			draw_command_data,
	CallNGMeshRenderBatches::RenderBatchesData::MeshShaderParams		mesh_shader_params,
	CallNGMeshRenderBatches::RenderBatchesData::MaterialShaderParams	mtl_shader_params)
{
	// Push back new RenderBatch object
	m_render_batches.push_back(RenderBatch());

	// Create shader program
	m_render_batches.back().shader_prgm = std::make_unique<GLSLShader>();
	vislib::graphics::gl::ShaderSource vert_shader_src;
	vislib::graphics::gl::ShaderSource frag_shader_src;
	// TODO get rid of vislib StringA...
	vislib::StringA shader_base_name(shader_prgm_data.raw_string);
	instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::vertex", vert_shader_src);
	instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::fragment", frag_shader_src);
	m_render_batches.back().shader_prgm->Create(vert_shader_src.Code(),vert_shader_src.Count(),frag_shader_src.Code(),frag_shader_src.Count());

	// Create mesh
	m_render_batches.back().mesh = std::make_unique<Mesh>(
		mesh_data.vertex_data.raw_data,
		mesh_data.vertex_data.byte_size,
		mesh_data.index_data.raw_data,
		mesh_data.index_data.byte_size,
		mesh_data.vertex_descriptor,
		mesh_data.index_data.index_type 
		);

	// Create GPU buffer for draw commands
	m_render_batches.back().draw_commands = std::make_unique<BufferObject>(
		GL_DRAW_INDIRECT_BUFFER,
		draw_command_data.data,
		draw_command_data.draw_cnt * sizeof(CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData::DrawElementsCommand),
		GL_DYNAMIC_DRAW
		);

	// Create GPU buffer for mesh related shader parameters
	m_render_batches.back().mesh_shader_params = std::make_unique<BufferObject>(
		GL_SHADER_STORAGE_BUFFER,
		mesh_shader_params.raw_data,
		mesh_shader_params.byte_size,
		GL_DYNAMIC_DRAW
		);

	// Create GPU buffer for material related shader parameters
	m_render_batches.back().mtl_shader_params = std::make_unique<BufferObject>(
		GL_SHADER_STORAGE_BUFFER,
		mtl_shader_params.data,
		mtl_shader_params.elements_cnt * sizeof(CallNGMeshRenderBatches::RenderBatchesData::MaterialParameters),
		GL_DYNAMIC_DRAW
		);
}

void NGMeshRenderer::updateRenderBatch(
	size_t																idx,
	CallNGMeshRenderBatches::RenderBatchesData::ShaderPrgmData			shader_prgm_data,
	CallNGMeshRenderBatches::RenderBatchesData::MeshData				mesh_data,
	CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData			draw_command_data,
	CallNGMeshRenderBatches::RenderBatchesData::MeshShaderParams		mesh_shader_params,
	CallNGMeshRenderBatches::RenderBatchesData::MaterialShaderParams	mtl_shader_params,
	uint32_t															update_flags)
{
	if (idx >= m_render_batches.size())
	{
		vislib::sys::Log::DefaultLog.WriteError("Invalid batch index for render batch update.");
		return;
	}

	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::shaderUpdateBit()) > 0)
	{
		m_render_batches[idx].shader_prgm.reset();

		m_render_batches[idx].shader_prgm = std::make_unique<GLSLShader>();
		vislib::graphics::gl::ShaderSource vert_shader_src;
		vislib::graphics::gl::ShaderSource frag_shader_src;
		// TODO get rid of vislib StringA...
		vislib::StringA shader_base_name(shader_prgm_data.raw_string);
		instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::vertex", vert_shader_src);
		instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::fragment", frag_shader_src);
		m_render_batches.back().shader_prgm->Create(vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
	}

	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::meshUpdateBit()) > 0)
	{
		// TODO check if mesh buffer need reallocation
	}

	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::drawCommandsUpdateBit()) > 0)
	{
		// TODO check if buffer needs reallocation
	}

	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::meshParamsUpdateBit()) > 0)
	{
		// TODO check if buffer needs reallocation
	}

	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::materialsParamsUpdateBit()) > 0)
	{
		// TODO check if buffer needs reallocation
	}
}

bool NGMeshRenderer::Render(megamol::core::Call& call)
{
	megamol::core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
	if (cr == NULL) return false;

	CallNGMeshRenderBatches* render_batch_call = this->getData();

	//TODO loop render batches data, create/update if necessary

	//vislib::sys::Log::DefaultLog.WriteError("Hey listen!");

	//TODO loop "registered" render batches
	for (auto& render_batch : m_render_batches)
	{
		render_batch.shader_prgm->Enable();

		render_batch.mesh_shader_params->bind(0);
		render_batch.mtl_shader_params->bind(1);

		render_batch.draw_commands->bind();
		render_batch.mesh->bindVertexArray();
		/*
		glMultiDrawElementsIndirect(render_batch.mesh->getPrimitiveType(),
			render_batch.mesh->getIndicesType(),
			(GLvoid*)0,
			render_batch.draw_cnt,
			0);
			*/
	}

	return true;
}