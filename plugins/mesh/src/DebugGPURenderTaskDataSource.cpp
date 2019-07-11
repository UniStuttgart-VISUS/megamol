/*
* DebugGPURenderTaskDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include <random>

#include "vislib/math/Matrix.h"

#include "DebugGPURenderTaskDataSource.h"

#include "mesh/CallGPURenderTaskData.h"
#include "mesh/CallGPUMaterialData.h"
#include "mesh/CallGPUMeshData.h"

megamol::mesh::DebugGPURenderTaskDataSource::DebugGPURenderTaskDataSource()
{
}

megamol::mesh::DebugGPURenderTaskDataSource::~DebugGPURenderTaskDataSource()
{
}

bool megamol::mesh::DebugGPURenderTaskDataSource::getDataCallback(core::Call & caller)
{
	CallGPURenderTaskData* rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
	if(rtc == NULL)
		return false;

	CallGPUMaterialData* mtlc = this->m_material_callerSlot.CallAs<CallGPUMaterialData>();
	if (mtlc == NULL)
		return false;

	if (!(*mtlc)(0))
		return false;

	CallGPUMeshData* mc = this->m_mesh_callerSlot.CallAs<CallGPUMeshData>();
	if (mc == NULL)
		return false;

	if (!(*mc)(0))
		return false;

	auto gpu_mtl_storage = mtlc->getMaterialStorage();
	auto gpu_mesh_storage = mc->getGPUMeshes();

	//TODO nullptr check

	std::mt19937 generator(m_gpu_render_tasks->getTotalDrawCount());
	std::uniform_real_distribution<float> distr(0.01f, 0.03f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData())
	{
		auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
		auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

		std::vector<DrawElementsCommand> draw_commands(1000, sub_mesh.sub_mesh_draw_command);

		std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR >> object_transform(1000);
		typedef std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR >> PerTaskData;

		for (int i = 0; i < 1000; ++i)
		{
			GLfloat scale = distr(generator);
			scale = 0.1f;
			object_transform[i].SetAt(0, 0, scale);
			object_transform[i].SetAt(1, 1, scale);
			object_transform[i].SetAt(2, 2, scale);

			object_transform[i].SetAt(3, 3, 1.0f);

			object_transform[i].SetAt(0, 3, loc_distr(generator));
			object_transform[i].SetAt(1, 3, loc_distr(generator));
			object_transform[i].SetAt(2, 3, loc_distr(generator));
		}

		m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, object_transform);

		//m_gpu_render_tasks->addSingleRenderTask<PerTaskData>(
		//	shader,
		//	gpu_batch_mesh,
		//	sub_mesh.sub_mesh_draw_command,
		//	object_transform
		//);
	}

	rtc->setRenderTaskData(m_gpu_render_tasks);

	return true;
}

bool megamol::mesh::DebugGPURenderTaskDataSource::load()
{

	//m_gpu_render_tasks->addSingleRenderTask()
	
	return false;
}
