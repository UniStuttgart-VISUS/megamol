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
	: m_render_task_data(std::make_shared<RenderTaskDataStorage>()), m_loaded(false)
{
}

megamol::ngmesh::DebugRenderTasksDataSource::~DebugRenderTasksDataSource()
{
}

bool megamol::ngmesh::DebugRenderTasksDataSource::getDataCallback(core::Call & caller)
{
	RenderTasksDataCall* task_call = dynamic_cast<RenderTasksDataCall*>(&caller);
	if (task_call == NULL)
		return false;

	if (!m_loaded)
	{
		load();

		task_call->setRenderTaskData(m_render_task_data);

		m_loaded = true;
	}

	return true;
}

bool megamol::ngmesh::DebugRenderTasksDataSource::load()
{
	std::mt19937 generator(42115);
	std::uniform_real_distribution<float> distr(0.01f, 0.03f);
	std::uniform_real_distribution<float> loc_distr(-0.9f, 0.9f);
	
	m_render_task_data->reserveBatch(0, 0, 1000000, 1000000 * 64);

	for (size_t i = 0; i < 1000000; ++i)
	{
		vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR > object_transform;

		GLfloat scale = distr(generator);
		scale = 0.1f;
		object_transform.SetAt(0, 0, scale);
		object_transform.SetAt(1, 1, scale);
		object_transform.SetAt(2, 2, scale);

		object_transform.SetAt(3, 3, 1.0f);
	
		object_transform.SetAt(0, 3, loc_distr(generator));
		object_transform.SetAt(1, 3, loc_distr(generator));
		object_transform.SetAt(2, 3, loc_distr(generator));
	
		m_render_task_data->addRenderTask<GLfloat*>(0, 0, 1, 0, { object_transform.PeekComponents(),object_transform.PeekComponents() + 16 });
	}

	return true;
}
