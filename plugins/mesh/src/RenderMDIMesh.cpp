/*
* RenderMDIMesh.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include <array>
#include <random>

#include "RenderMDIMesh.h"

#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/ShaderSource.h"

#include "mesh/CallGPUMeshData.h"
#include "mesh/CallGPUMaterialData.h"
#include "mesh/CallGPURenderTaskData.h"

using namespace megamol::core;
using namespace megamol::mesh;

RenderMDIMesh::RenderMDIMesh()
	: Renderer3DModule(),
	m_render_task_callerSlot("getRenderTaskData", "Connects the renderer with a render task data source")
{
	this->m_render_task_callerSlot.SetCompatibleCall<GPURenderTasksDataCallDescription>();
	this->MakeSlotAvailable(&this->m_render_task_callerSlot);
}

RenderMDIMesh::~RenderMDIMesh()
{
	this->Release();
}

bool RenderMDIMesh::create()
{
	// generate debug render batch that doesn't rely on data call
	/*
	CallmeshRenderBatches::RenderBatchesData::ShaderPrgmData			shader_prgm_data;
	CallmeshRenderBatches::RenderBatchesData::MeshData				mesh_data;
	CallmeshRenderBatches::RenderBatchesData::DrawCommandData			draw_command_data;
	CallmeshRenderBatches::RenderBatchesData::MeshShaderParams		obj_shader_params;
	CallmeshRenderBatches::RenderBatchesData::MaterialShaderParams	mtl_shader_params;

	shader_prgm_data.raw_string = "meshDebug";
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
	draw_command_data.data = new CallmeshRenderBatches::RenderBatchesData::DrawCommandData::DrawElementsCommand[draw_command_data.draw_cnt];

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

void RenderMDIMesh::release()
{
	m_per_frame_data.reset();
}

bool RenderMDIMesh::GetExtents(megamol::core::Call& call)
{
	view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
	if (cr == NULL)
		return false;

	CallGPURenderTaskData* rtc = this->m_render_task_callerSlot.CallAs<CallGPURenderTaskData>();
	
	if (rtc == NULL)
		return false;
	
	if (!(*rtc)(1))
		return false;

	cr->SetTimeFramesCount(rtc->FrameCount());
	cr->AccessBoundingBoxes() = rtc->GetBoundingBoxes();
	cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

	return true;
}

bool RenderMDIMesh::Render(megamol::core::Call& call)
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
	
	CallGPURenderTaskData* task_call = this->m_render_task_callerSlot.CallAs<CallGPURenderTaskData>();
	
	if (task_call == NULL)
		return false;
	
	if ((!(*task_call)(0)) )
		return false;
	
	//vislib::sys::Log::DefaultLog.WriteError("Hey listen!");
	
	// Set GL state (otherwise bounding box or view cube rendering state is used)
	//glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
    glDisable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	auto gpu_render_tasks = task_call->getRenderTaskData();

	auto const& per_frame_buffers = gpu_render_tasks->getPerFrameBuffers();

	for (auto const& buffer : per_frame_buffers)
	{
        std::get<0>(buffer)->bind(std::get<1>(buffer));
	}
	
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

		//CallmeshRenderBatches::RenderBatchesData::DrawCommandData::DrawElementsCommand command_buffer;
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