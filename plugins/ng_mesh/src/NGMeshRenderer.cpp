/*
* NGMeshRenderer.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include <array>
#include <random>

#include "NGMeshRenderer.h"

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include "ng_mesh/GPUMeshDataCall.h"
#include "ng_mesh/GPUMaterialDataCall.h"
#include "ng_mesh/GPURenderTaskDataCall.h"

using namespace megamol::core;
using namespace megamol::ngmesh;

NGMeshRenderer::NGMeshRenderer()
	: Renderer3DModule(),
	//m_renderBatches_callerSlot("getData", "Connects the mesh renderer with a mesh data source"),
	//m_mesh_callerSlot("getMeshData", "Connects the renderer with a mesh data source"),
	//m_material_callerSlot("getMaterialData", "Connects the renderer with a material data source"),
	m_render_task_callerSlot("getRenderTaskData", "Connects the renderer with a render task data source")
{
	//this->m_renderBatches_callerSlot.SetCompatibleCall<CallNGMeshRenderBatchesDescription>();
	//this->MakeSlotAvailable(&this->m_renderBatches_callerSlot);

	//this->m_mesh_callerSlot.SetCompatibleCall<GPUMeshDataCallDescription>();
	//this->MakeSlotAvailable(&this->m_mesh_callerSlot);
	//
	//this->m_material_callerSlot.SetCompatibleCall<GPUMaterialDataCallDescription>();
	//this->MakeSlotAvailable(&this->m_material_callerSlot);

	this->m_render_task_callerSlot.SetCompatibleCall<GPURenderTasksDataCallDescription>();
	this->MakeSlotAvailable(&this->m_render_task_callerSlot);
}

NGMeshRenderer::~NGMeshRenderer()
{
	this->Release();
}

bool NGMeshRenderer::create()
{
	// generate debug render batch that doesn't rely on data call
	/*
	CallNGMeshRenderBatches::RenderBatchesData::ShaderPrgmData			shader_prgm_data;
	CallNGMeshRenderBatches::RenderBatchesData::MeshData				mesh_data;
	CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData			draw_command_data;
	CallNGMeshRenderBatches::RenderBatchesData::MeshShaderParams		obj_shader_params;
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

	std::mt19937 generator(4215);
	std::uniform_real_distribution<float> distr(0.05, 0.1);
	std::uniform_real_distribution<float> loc_distr(-0.9, 0.9);

	draw_command_data.draw_cnt = 1000000;
	draw_command_data.data = new CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData::DrawElementsCommand[draw_command_data.draw_cnt];

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

		std::memcpy(obj_shader_params.raw_data + i*(16*4), object_transform.PeekComponents(), 16*4);
	}

	mtl_shader_params.elements_cnt = 0;
	
	addRenderBatch(shader_prgm_data, mesh_data, draw_command_data, obj_shader_params, mtl_shader_params);
	
	//TODO delete stuff again
	*/

	return true;
}

void NGMeshRenderer::release()
{
	//m_render_batches.clear();
	m_per_frame_data.reset();
}

bool NGMeshRenderer::GetExtents(megamol::core::Call& call)
{
	view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
	if (cr == NULL)
		return false;

	//CallNGMeshRenderBatches* render_batch_call = this->m_renderBatches_callerSlot.CallAs<CallNGMeshRenderBatches>();
	//
	//if (render_batch_call == NULL)
	//	return false;
	//
	//if (!(*render_batch_call)(1))
	//	return false;
	//
	//cr->SetTimeFramesCount(render_batch_call->FrameCount());
	//cr->AccessBoundingBoxes() = render_batch_call->GetBoundingBoxes();
	//cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);


	GPURenderTaskDataCall* rtc = this->m_render_task_callerSlot.CallAs<GPURenderTaskDataCall>();
	
	if (rtc == NULL)
		return false;
	
	if (!(*rtc)(1))
		return false;

	cr->SetTimeFramesCount(rtc->FrameCount());
	cr->AccessBoundingBoxes() = rtc->GetBoundingBoxes();
	cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

	return true;
}

//   void NGMeshRenderer::addRenderBatch(
//   	ShaderPrgmDataAccessor const&			shader_prgm_data,
//   	MeshDataAccessor const&					mesh_data,
//   	DrawCommandDataAccessor const&			draw_command_data,
//   	ObjectShaderParamsDataAccessor const&	obj_shader_params,
//   	MaterialShaderParamsDataAccessor const&	mtl_shader_params)
//   {
//   	// Push back new RenderBatch object
//   	m_render_batches.push_back(RenderBatch());
//   
//   	// Create shader program
//   	m_render_batches.back().shader_prgm = std::make_unique<GLSLShader>();
//   	vislib::graphics::gl::ShaderSource vert_shader_src;
//   	vislib::graphics::gl::ShaderSource frag_shader_src;
//   	// TODO get rid of vislib StringA...
//   	vislib::StringA shader_base_name(shader_prgm_data.raw_string);
//   	instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::vertex", vert_shader_src);
//   	instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::fragment", frag_shader_src);
//   	m_render_batches.back().shader_prgm->Create(vert_shader_src.Code(),vert_shader_src.Count(),frag_shader_src.Code(),frag_shader_src.Count());
//   
//   	// Create mesh
//   	VertexLayout layout;
//   	layout.stride = mesh_data.vertex_descriptor.stride;
//   	for (size_t i = 0; i < mesh_data.vertex_descriptor.attribute_cnt; ++i)
//   	{
//   		layout.attributes.push_back(VertexLayout::Attribute(
//   			mesh_data.vertex_descriptor.attributes[i].size,
//   			mesh_data.vertex_descriptor.attributes[i].type,
//   			mesh_data.vertex_descriptor.attributes[i].normalized,
//   			mesh_data.vertex_descriptor.attributes[i].offset)
//   		);
//   	}
//   
//   	if (mesh_data.vertex_data.buffer_cnt > 0 && mesh_data.vertex_data.buffer_cnt == mesh_data.vertex_descriptor.attribute_cnt)
//   	{
//   		std::vector<uint8_t*> vertex_data_ptrs(mesh_data.vertex_data.buffer_cnt);
//   		std::vector<size_t> vertex_data_sizes(mesh_data.vertex_data.buffer_cnt);
//   
//   		for (int i = 0; i< mesh_data.vertex_data.buffer_cnt; ++i)
//   		{
//   			vertex_data_ptrs[i] = mesh_data.vertex_data.buffers[i].raw_data;
//   			vertex_data_sizes[i] = mesh_data.vertex_data.buffers[i].byte_size;
//   
//   			//std::cout << "Vertex Data Pointer Offset: " << uint32_view[i] << std::endl;
			//std::cout << "Vertex Data Sizes: " << vertex_data_sizes[i] << std::endl;
//   		}
//   		
//   		m_render_batches.back().mesh = std::make_unique<Mesh>(
//   			vertex_data_ptrs,
//   			vertex_data_sizes,
//   			mesh_data.index_data.raw_data,
//   			mesh_data.index_data.byte_size,
//   			layout,
//   			mesh_data.index_data.index_type,
//   			mesh_data.usage,
//   			mesh_data.primitive_type
//   			);
//   	}
//   	else if (mesh_data.vertex_data.buffer_cnt == 1) // buffer_cnt != attrib_cnt signals one vertex buffer for all attributes
//   	{
//   		m_render_batches.back().mesh = std::make_unique<Mesh>(
//   			mesh_data.vertex_data.buffers[0].raw_data,
//   			mesh_data.vertex_data.buffers[0].byte_size,
//   			mesh_data.index_data.raw_data,
//   			mesh_data.index_data.byte_size,
//   			layout,
//   			mesh_data.index_data.index_type,
//   			mesh_data.usage,
//   			mesh_data.primitive_type
//   			);
//   	}
//   	else
//   	{
//   		//fail?
//   	}
//   
//   	// Create GPU buffer for draw commands
//   	m_render_batches.back().draw_commands = std::make_unique<BufferObject>(
//   		GL_DRAW_INDIRECT_BUFFER,
//   		draw_command_data.data,
//   		draw_command_data.draw_cnt * sizeof(DrawCommandDataAccessor::DrawElementsCommand),
//   		GL_DYNAMIC_DRAW
//   		);
//   
//   	m_render_batches.back().draw_cnt = draw_command_data.draw_cnt;
//   
//   	// Create GPU buffer for mesh related shader parameters
//   	m_render_batches.back().obj_shader_params = std::make_unique<BufferObject>(
//   		GL_SHADER_STORAGE_BUFFER,
//   		obj_shader_params.raw_data,
//   		obj_shader_params.byte_size,
//   		GL_DYNAMIC_DRAW
//   		);
//   
//   	// Create GPU buffer for material related shader parameters
	//TODO build textures from input?
//   	m_render_batches.back().mtl_shader_params = std::make_unique<BufferObject>(
//   		GL_SHADER_STORAGE_BUFFER,
//   		mtl_shader_params.data,
//   		mtl_shader_params.elements_cnt * sizeof(MaterialParameters),
//   		GL_DYNAMIC_DRAW
//   		);
//   
//   	glMemoryBarrier(GL_ALL_BARRIER_BITS);
//   }

//   void NGMeshRenderer::updateRenderBatch(
//   	size_t									idx,
//   	ShaderPrgmDataAccessor const&			shader_prgm_data,
//   	MeshDataAccessor const&					mesh_data,
//   	DrawCommandDataAccessor const&			draw_command_data,
//   	ObjectShaderParamsDataAccessor const&	obj_shader_params,
//   	MaterialShaderParamsDataAccessor const&	mtl_shader_params,
//   	uint32_t								update_flags)
//   {
//   	if (idx > m_render_batches.size())
//   	{
//   		vislib::sys::Log::DefaultLog.WriteError("Invalid batch index for render batch update.");
//   		return;
//   	}
//   	else if (idx == m_render_batches.size())
//   	{
//   		vislib::sys::Log::DefaultLog.WriteInfo("Creating new GPU render batch.");
//   		addRenderBatch(shader_prgm_data,
//   			mesh_data,
//   			draw_command_data,
//   			obj_shader_params,
//   			mtl_shader_params);
//   
//   		return;
//   	}
//   
//   	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::UpdateBits::SHADER_BIT) > 0)
//   	{
//   		m_render_batches[idx].shader_prgm.reset();
//   
//   		m_render_batches[idx].shader_prgm = std::make_unique<GLSLShader>();
//   		vislib::graphics::gl::ShaderSource vert_shader_src;
//   		vislib::graphics::gl::ShaderSource frag_shader_src;
//   		// TODO get rid of vislib StringA...
//   		vislib::StringA shader_base_name(shader_prgm_data.raw_string);
//   		instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::vertex", vert_shader_src);
//   		instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::fragment", frag_shader_src);
//   		m_render_batches.back().shader_prgm->Create(vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
//   	}
//   
//   	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::UpdateBits::MESH_BIT) > 0)
//   	{
//   		m_render_batches[idx].mesh.reset();
//   
//   		VertexLayout layout;
//   		layout.stride = mesh_data.vertex_descriptor.stride;
//   		for (size_t i = 0; i < mesh_data.vertex_descriptor.attribute_cnt; ++i)
//   		{
//   			layout.attributes.push_back(VertexLayout::Attribute(
//   				mesh_data.vertex_descriptor.attributes[i].size,
//   				mesh_data.vertex_descriptor.attributes[i].type,
//   				mesh_data.vertex_descriptor.attributes[i].normalized,
//   				mesh_data.vertex_descriptor.attributes[i].offset)
//   			);
//   		}
//   
//   		std::vector<uint8_t*> vertex_data_ptrs(mesh_data.vertex_data.buffer_cnt);
//   		std::vector<size_t> vertex_data_sizes(mesh_data.vertex_data.buffer_cnt);
//   
//   		for (int i = 0; i< mesh_data.vertex_data.buffer_cnt; ++i)
//   		{
//   			vertex_data_ptrs[i] = mesh_data.vertex_data.buffers[i].raw_data;
//   			vertex_data_sizes[i] = mesh_data.vertex_data.buffers[i].byte_size;
//   
//   			//std::cout << "Vertex Data Pointer Offset: " << uint32_view[i] << std::endl;
			//std::cout << "Vertex Data Sizes: " << vertex_data_sizes[i] << std::endl;
//   		}
//   
//   		m_render_batches[idx].mesh = std::make_unique<Mesh>(
//   			vertex_data_ptrs,
//   			vertex_data_sizes,
//   			mesh_data.index_data.raw_data,
//   			mesh_data.index_data.byte_size,
//   			layout,
//   			mesh_data.index_data.index_type
//   			);
//   	}
//   
//   	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::UpdateBits::DRAWCOMMANDS_BIT) > 0)
//   	{
//   		m_render_batches[idx].draw_commands.reset();
//   
//   		// Create GPU buffer for draw commands
//   		m_render_batches[idx].draw_commands = std::make_unique<BufferObject>(
//   			GL_DRAW_INDIRECT_BUFFER,
//   			draw_command_data.data,
//   			draw_command_data.draw_cnt * sizeof(DrawCommandDataAccessor::DrawElementsCommand),
//   			GL_DYNAMIC_DRAW
//   			);
//   
//   		m_render_batches.back().draw_cnt = draw_command_data.draw_cnt;
//   	}
//   
//   	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::UpdateBits::MESHPARAMS_BIT) > 0)
//   	{
//   		m_render_batches[idx].obj_shader_params.reset();
//   
//   		// Create GPU buffer for mesh related shader parameters
//   		m_render_batches[idx].obj_shader_params = std::make_unique<BufferObject>(
//   			GL_SHADER_STORAGE_BUFFER,
//   			obj_shader_params.raw_data,
//   			obj_shader_params.byte_size,
//   			GL_DYNAMIC_DRAW
//   			);
//   	}
//   
//   	if ((update_flags & CallNGMeshRenderBatches::RenderBatchesData::UpdateBits::MATERIAL_BIT) > 0)
//   	{
//   		m_render_batches[idx].mtl_shader_params.reset();
//   
//   		// Create GPU buffer for material related shader parameters
		//TODO build textures from input?
//   		m_render_batches[idx].mtl_shader_params = std::make_unique<BufferObject>(
//   			GL_SHADER_STORAGE_BUFFER,
//   			mtl_shader_params.data,
//   			mtl_shader_params.elements_cnt * sizeof(MaterialParameters),
//   			GL_DYNAMIC_DRAW
//   			);
//   	}
//   
//   	glMemoryBarrier(GL_ALL_BARRIER_BITS);
//   }
//   

//void megamol::ngmesh::NGMeshRenderer::updateMeshes(BatchedMeshesDataAccessor const & meshes, uint32_t update_flags)
//{
//	if (update_flags == UPDATE_ALL_BIT)
//	{
//		m_meshes.clear();
//		m_meshes.reserve(meshes.batch_cnt);
//
//		for (size_t i = 0; i < meshes.batch_cnt; i++)
//		{
//			auto mesh = createMesh(meshes, i);
//
//			m_meshes.push_back(BatchedMeshes());
//			m_meshes.back().mesh = mesh;
//
//			m_meshes.back().submesh_draw_commands.resize(meshes.draw_command_batches[i].draw_cnt);
//			auto dest = m_meshes.back().submesh_draw_commands.data();
//			auto src_first = meshes.draw_command_batches[i].draw_commands;
//			auto src_last = meshes.draw_command_batches[i].draw_commands + meshes.draw_command_batches[i].draw_cnt;
//			std::copy(src_first, src_last, dest);
//		}
//	}
//	else if (update_flags == UPDATE_INDVIDUAL_BIT)
//	{
//		for (size_t i = 0; i < meshes.batch_cnt; i++)
//		{
//			if (/* update for batch required*/true)
//			{
//				auto mesh = createMesh(meshes, i);
//				m_meshes[i].mesh = mesh;
//
//				m_meshes.back().submesh_draw_commands.resize(meshes.draw_command_batches[i].draw_cnt);
//				auto dest = m_meshes.back().submesh_draw_commands.data();
//				auto src_first = meshes.draw_command_batches[i].draw_commands;
//				auto src_last = meshes.draw_command_batches[i].draw_commands + meshes.draw_command_batches[i].draw_cnt;
//				std::copy(src_first, src_last, dest);
//			}
//		}
//	}
//	else if (update_flags == DATA_ADDED_BIT)
//	{
//		size_t start_idx = m_meshes.size();
//
//		for (size_t i = start_idx; i < meshes.batch_cnt; i++)
//		{
//			auto mesh = createMesh(meshes, i);
//
//			m_meshes.push_back(BatchedMeshes());
//			m_meshes.back().mesh = mesh;
//
//			m_meshes.back().submesh_draw_commands.resize(meshes.draw_command_batches[i].draw_cnt);
//			auto dest = m_meshes.back().submesh_draw_commands.data();
//			auto src_first = meshes.draw_command_batches[i].draw_commands;
//			auto src_last = meshes.draw_command_batches[i].draw_commands + meshes.draw_command_batches[i].draw_cnt;
//			std::copy(src_first, src_last, dest);
//		}
//	}
//}
//
//void megamol::ngmesh::NGMeshRenderer::updateMaterials(std::shared_ptr<MaterialsDataStorage> const & materials, uint32_t update_flags)
//{
//	for (auto& material : materials->m_materials)
//	{
//		// Create shader program
//		m_shader_programs.push_back(std::make_unique<GLSLShader>());
//
//		vislib::graphics::gl::ShaderSource vert_shader_src;
//		vislib::graphics::gl::ShaderSource frag_shader_src;
//		// TODO get rid of vislib StringA...
//		vislib::StringA shader_base_name(material.btf_filename.c_str());
//		instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::vertex", vert_shader_src);
//		instance()->ShaderSourceFactory().MakeShaderSource(shader_base_name + "::fragment", frag_shader_src);
//
//		m_shader_programs.back()->Create(vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
//
//		m_materials.push_back(Material());
//		m_materials.back().btf_name = material.btf_filename;
//		m_materials.back().shader = m_shader_programs.back();
//	}
//}
//
//void megamol::ngmesh::NGMeshRenderer::updateRenderTasks(std::shared_ptr<RenderTaskDataStorage> const & render_tasks, uint32_t update_flags)
//{
//	for (auto& task_batch : render_tasks->m_batched_render_task)
//	{
//		m_render_batches.push_back(RenderBatch());
//		RenderBatch& render_batch = m_render_batches.back();
//
//		render_batch.draw_cnt = task_batch.total_draw_cnt;
//		render_batch.shader_prgm = m_materials.at(task_batch.material_idx).shader;
//
//		size_t mesh_batch_idx = 0/* task_batch.mesh_idx*/;
//
//		render_batch.mesh = m_meshes.at(mesh_batch_idx).mesh;
//
//		// Create GPU buffer for per object shader parameters
//	 	m_render_batches.back().obj_shader_params = std::make_unique<BufferObject>(
//	 		GL_SHADER_STORAGE_BUFFER,
//			task_batch.per_object_data,
//	 		GL_DYNAMIC_DRAW);
//
//		//TODO material parameter for shader
//		std::array<uint64_t, 4> dummy_texture_handles;
//		m_render_batches.back().mtl_shader_params = std::make_unique<BufferObject>(
//			GL_SHADER_STORAGE_BUFFER,
//			dummy_texture_handles,
//			GL_DYNAMIC_DRAW);
//
//		std::vector<DrawElementsCommand> draw_command_data;
//		draw_command_data.reserve(render_batch.draw_cnt);
//		for (auto& task : task_batch.render_tasks)
//		{
//			size_t base_idx = task.draw_commands_base_offset;
//			for (size_t i = 0; i < task.draw_commands_cnt; i++)
//			{
//				draw_command_data.push_back(m_meshes[task_batch.mesh_idx].submesh_draw_commands[base_idx + i]);
//			}
//		}
//
//		// Create GPU buffer for draw commands
//	 	m_render_batches.back().draw_commands = std::make_unique<BufferObject>(
//	 		GL_DRAW_INDIRECT_BUFFER,
//	 		draw_command_data,
//	 		GL_DYNAMIC_DRAW
//	 		);
//	}
//}


bool NGMeshRenderer::Render(megamol::core::Call& call)
{
	
	megamol::core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
	if (cr == NULL) return false;
	
	// manual creation of projection and view matrix
	GLfloat fovy = (cr->GetCameraParameters()->ApertureAngle() / 180.0f) * 3.14f;
	GLfloat near_clip = cr->GetCameraParameters()->NearClip();
	GLfloat far_clip = cr->GetCameraParameters()->FarClip();
	GLfloat f = 1.0f / std::tan(fovy / 2.0f);
	GLfloat nf = 1.0f / (near_clip - far_clip);
	GLfloat aspect_ratio = static_cast<GLfloat>(cr->GetViewport().AspectRatio());
	std::array<GLfloat, 16> projection_matrix;
	projection_matrix[0] = f / aspect_ratio;
	projection_matrix[1] = 0.0f;
	projection_matrix[2] = 0.0f;
	projection_matrix[3] = 0.0f;
	projection_matrix[4] = 0.0f;
	projection_matrix[5] = f;
	projection_matrix[6] = 0.0f;
	projection_matrix[7] = 0.0f;
	projection_matrix[8] = 0.0f;
	projection_matrix[9] = 0.0f;
	projection_matrix[10] = (far_clip + near_clip) * nf;
	projection_matrix[11] = -1.0f;
	projection_matrix[12] = 0.0f;
	projection_matrix[13] = 0.0f;
	projection_matrix[14] = (2.0f * far_clip * near_clip) * nf;
	projection_matrix[15] = 0.0f;
	
	auto cam_right = cr->GetCameraParameters()->Right();
	auto cam_up = cr->GetCameraParameters()->Up();
	auto cam_front = -cr->GetCameraParameters()->Front();
	auto cam_position = cr->GetCameraParameters()->Position();
	std::array<GLfloat, 16> view_matrix;
	view_matrix[0] = cam_right.X();
	view_matrix[1] = cam_up.X();
	view_matrix[2] = cam_front.X();
	view_matrix[3] = 0.0f;
	
	view_matrix[4] = cam_right.Y();
	view_matrix[5] = cam_up.Y();
	view_matrix[6] = cam_front.Y();
	view_matrix[7] = 0.0f;
	
	view_matrix[8] = cam_right.Z();
	view_matrix[9] = cam_up.Z();
	view_matrix[10] = cam_front.Z();
	view_matrix[11] = 0.0f;
	
	view_matrix[12] = - (cam_position.X()*cam_right.X() + cam_position.Y()*cam_right.Y() + cam_position.Z()*cam_right.Z());
	view_matrix[13] = - (cam_position.X()*cam_up.X() + cam_position.Y()*cam_up.Y() + cam_position.Z()*cam_up.Z());
	view_matrix[14] = - (cam_position.X()*cam_front.X() + cam_position.Y()*cam_front.Y() + cam_position.Z()*cam_front.Z());
	view_matrix[15] = 1.0f;
	
	// this is the apex of suck and must die
    glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix.data());
    glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix.data());
	
	//GPUMeshDataCall*       mesh_call = this->m_mesh_callerSlot.CallAs<GPUMeshDataCall>();
	//GPUMaterialDataCall*   matl_call = this->m_material_callerSlot.CallAs<GPUMaterialDataCall>();
	GPURenderTaskDataCall* task_call = this->m_render_task_callerSlot.CallAs<GPURenderTaskDataCall>();
	
	if (/*mesh_call == NULL || matl_call == NULL ||*/ task_call == NULL)
		return false;
	
	if ( /*(!(*mesh_call)(0)) || (!(*matl_call)(0)) ||*/ (!(*task_call)(0)) )
		return false;
		
	//	if (mesh_call->getUpdateFlags() > 0) {
	//		updateMeshes(*(mesh_call->getBatchedMeshesDataAccessor()), mesh_call->getUpdateFlags());
	//		mesh_call->resetUpdateFlags();
	//	}
	//	
	//	if (matl_call->getUpdateFlags() > 0) {
	//		updateMaterials(matl_call->getMaterialsData(), matl_call->getUpdateFlags());
	//		matl_call->resetUpdateFlags();
	//	}
	//	
	//	if (task_call->getUpdateFlags() > 0) {
	//		updateRenderTasks(task_call->getRenderTaskData(), task_call->getUpdateFlags());
	//		task_call->resetUpdateFlags();
	//	}
	
	//   CallNGMeshRenderBatches* render_batch_call = this->m_renderBatches_callerSlot.CallAs<CallNGMeshRenderBatches>();
	//   
	//   if (render_batch_call == NULL)
	//   	return false;
	//   
	//   if (!(*render_batch_call)(0))
	//   	return false;
	//   
	//   // loop through render batches data, update GPU render batches if necessary
	//   auto render_batches = render_batch_call->getRenderBatches();
	//   
	//   if (render_batches == nullptr)
	//   	return true;
	//   
	//   for (size_t batch_idx = 0; batch_idx < render_batches->getBatchCount(); ++batch_idx)
	//   {
	//   	if (render_batches->getUpdateFlags(batch_idx) > 0) // check if at least a single flag is set to 1
	//   	{
	//   		updateRenderBatch(
	//   			batch_idx,
	//   			render_batches->getShaderProgramData(batch_idx),
	//   			render_batches->getMeshData(batch_idx),
	//   			render_batches->getDrawCommandData(batch_idx),
	//   			render_batches->getObjectShaderParams(batch_idx),
	//   			render_batches->getMaterialShaderParams(batch_idx),
	//   			render_batches->getUpdateFlags(batch_idx)
	//   		);
	//   
	//   		render_batches->resetUpdateFlags(batch_idx);
	//   	}
	//   }
	
	// TODO update data from calls
	
	//vislib::sys::Log::DefaultLog.WriteError("Hey listen!");
	
	// Set GL state (otherwise bounding box or view cube rendering state is used)
	//glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
    glDisable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	auto gpu_render_tasks = task_call->getRenderTaskData();
	
	// loop through "registered" render batches
	for (auto const& render_task : gpu_render_tasks->getRenderTasks())
	{
		render_task.shader_program->Enable();
		
		// TODO introduce per frame "global" data buffer to store information like camera matrices?
		glUniformMatrix4fv(render_task.shader_program->ParameterLocation("view_mx"), 1, GL_FALSE, view_matrix.data());
		glUniformMatrix4fv(render_task.shader_program->ParameterLocation("proj_mx"), 1, GL_FALSE, projection_matrix.data());
		
		render_task.per_draw_data->bind(0);
		
		render_task.draw_commands->bind();
		render_task.mesh->bindVertexArray();
		
		glMultiDrawElementsIndirect(render_task.mesh->getPrimitiveType(),
			render_task.mesh->getIndicesType(),
			(GLvoid*)0,
			render_task.draw_cnt,
			0);

		//CallNGMeshRenderBatches::RenderBatchesData::DrawCommandData::DrawElementsCommand command_buffer;
		//command_buffer.cnt = 3;
		//command_buffer.instance_cnt = 1;
		//command_buffer.first_idx = 0;
		//command_buffer.base_vertex = 0;
		//command_buffer.base_instance = 0;

		//DrawElementsCommand command_buffer;
		//command_buffer.cnt = 3;
		//command_buffer.instance_cnt = 1;
		//command_buffer.first_idx = 0;
		//command_buffer.base_vertex = 0;
		//command_buffer.base_instance = 0;
		//
		//glDrawElementsIndirect(render_batch.mesh->getPrimitiveType(),
		//	render_batch.mesh->getIndicesType(),
		//	&command_buffer);

		//GLenum err = glGetError();
		//std::cout << "Error: " << err << std::endl;
	}
	
	// Clear the way for his ancient majesty, the mighty immediate mode...
	glUseProgram(0);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
	
	return true;
}

std::shared_ptr<Mesh> megamol::ngmesh::NGMeshRenderer::createMesh(BatchedMeshesDataAccessor const & meshes, size_t mesh_idx)
{
	std::shared_ptr<Mesh> retval(nullptr);

	MeshDataAccessor& mesh_data = meshes.mesh_data_batches[mesh_idx];

	VertexLayout layout;
	layout.stride = mesh_data.vertex_stride;
	for (size_t j = 0; j < mesh_data.vertex_attribute_cnt; j++)
	{
		layout.attributes.push_back(mesh_data.vertex_attributes[j]);
	}

	if (mesh_data.vertex_buffer_cnt > 0 && mesh_data.vertex_buffer_cnt == mesh_data.vertex_attribute_cnt)
	{
		std::vector<GLvoid*> vertex_data_ptrs(mesh_data.vertex_buffer_cnt);
		std::vector<size_t> vertex_data_sizes(mesh_data.vertex_buffer_cnt);

		for (int j = 0; j < mesh_data.vertex_buffer_cnt; ++j)
		{
			vertex_data_ptrs[j] = meshes.buffer_accessors[mesh_data.vertex_buffers_accessors_base_index + j].raw_data;
			vertex_data_sizes[j] = meshes.buffer_accessors[mesh_data.vertex_buffers_accessors_base_index + j].byte_size;

			//std::cout << "Vertex Data Pointer Offset: " << uint32_view[i] << std::endl;
			//std::cout << "Vertex Data Sizes: " << vertex_data_sizes[i] << std::endl;
		}

		std::byte* index_data_ptr = meshes.buffer_accessors[mesh_data.index_buffer_accessor_index].raw_data;
		size_t index_data_size = meshes.buffer_accessors[mesh_data.index_buffer_accessor_index].byte_size;

		retval = std::make_shared<Mesh>(
			vertex_data_ptrs,
			vertex_data_sizes,
			index_data_ptr,
			index_data_size,
			layout,
			mesh_data.index_type,
			mesh_data.usage,
			mesh_data.primitive_type);
	}
	else if (mesh_data.vertex_buffer_cnt == 1) // buffer_cnt != attrib_cnt signals one vertex buffer for all attributes
	{
		retval = std::make_shared<Mesh>(
			meshes.buffer_accessors[mesh_data.vertex_buffers_accessors_base_index].raw_data,
			meshes.buffer_accessors[mesh_data.vertex_buffers_accessors_base_index].byte_size,
			meshes.buffer_accessors[mesh_data.index_buffer_accessor_index].raw_data,
			meshes.buffer_accessors[mesh_data.index_buffer_accessor_index].byte_size,
			layout,
			mesh_data.index_type,
			mesh_data.usage,
			mesh_data.primitive_type
			);
	}
	else
	{
		//fail?
	}

	return retval;
}
