/*
* DebugRenderTasksDataSource.cpp
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#include "stdafx.h"

#include <vector>
#include <random>
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "DebugRenderTasksDataSource.h"

#include "ng_mesh/RenderTasksDataCall.h"

megamol::ngmesh::DebugRenderTasksDataSource::DebugRenderTasksDataSource()
{
}

megamol::ngmesh::DebugRenderTasksDataSource::~DebugRenderTasksDataSource()
{
}

bool megamol::ngmesh::DebugRenderTasksDataSource::getDataCallback(core::Call & caller)
{
	RenderTasksDataCall* material_call = dynamic_cast<RenderTasksDataCall*>(&caller);
	if (material_call == NULL)
		return false;

	load();

	//TODO hand over data access to call

	return true;
}

bool megamol::ngmesh::DebugRenderTasksDataSource::load()
{
	std::mt19937 generator(4215);
	std::uniform_real_distribution<float> distr(0.01f, 0.03f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);

	std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> object_transforms(100);
	//	obj_shader_params.byte_size = object_transforms.size() * sizeof(vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>);
	//	obj_shader_params.raw_data = reinterpret_cast<uint8_t*>(object_transforms.data());
	//	
	//	for (int i = 0; i < 100; ++i)
	//	{
	//		draw_command_data.data[i].cnt = 3;
	//		draw_command_data.data[i].instance_cnt = 1;
	//		draw_command_data.data[i].first_idx = 0;
	//		draw_command_data.data[i].base_vertex = 0;
	//		draw_command_data.data[i].base_instance = 0;
	//	
	//		GLfloat scale = distr(generator);
	//		scale = 1.0f;
	//		object_transforms[i].SetAt(0, 0, scale);
	//		object_transforms[i].SetAt(1, 1, scale);
	//		object_transforms[i].SetAt(2, 2, scale);
	//	
	//		object_transforms[i].SetAt(0, 3, loc_distr(generator));
	//		object_transforms[i].SetAt(1, 3, loc_distr(generator));
	//		object_transforms[i].SetAt(2, 3, loc_distr(generator));
	//	
	//		m_render_task_data.addRenderTask(0, 0, 1, 0, )
	//	}

	return true;
}
