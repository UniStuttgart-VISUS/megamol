#include "stdafx.h"

#include "glTFRenderTasksDataSource.h"
#include "tiny_gltf.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "ng_mesh/RenderTasksDataCall.h"

megamol::ngmesh::GlTFRenderTasksDataSource::GlTFRenderTasksDataSource()
	: m_glTF_callerSlot("getGlTFFile", "Connects the data source with a loaded glTF file")
{
	this->m_glTF_callerSlot.SetCompatibleCall<GlTFDataCallDescription>();
	this->MakeSlotAvailable(&this->m_glTF_callerSlot);
}

megamol::ngmesh::GlTFRenderTasksDataSource::~GlTFRenderTasksDataSource()
{
}

bool megamol::ngmesh::GlTFRenderTasksDataSource::getDataCallback(core::Call & caller)
{
	RenderTasksDataCall* mesh_call = dynamic_cast<RenderTasksDataCall*>(&caller);
	if (mesh_call == NULL)
		return false;

	GlTFDataCall* gltf_call = this->m_glTF_callerSlot.CallAs<GlTFDataCall>();
	if (gltf_call == NULL)
		return false;

	if (!(*gltf_call)(0))
		return false;

	if (gltf_call->getUpdateFlag())
	{
		auto model = gltf_call->getGlTFModel();

		// reserve enough room to theoretically store a render task per node
		m_render_task_data->reserveBatch(0, 0, model->nodes.size(), model->nodes.size() * 64);

		for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++)
		{
			if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1)
			{
				vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR > object_transform;

				if (model->nodes[node_idx].matrix.size() != 0) // has matrix transform
				{
					// TODO
				}
				else
				{
					auto& translation = model->nodes[node_idx].translation;
					auto& scale = model->nodes[node_idx].scale;
					auto& rotation = model->nodes[node_idx].rotation;

					if (translation.size() != 0) {
						
					}

					if (scale.size() != 0) {

					}

					if (rotation.size() != 0) {

					}
				}

				m_render_task_data->addRenderTask<GLfloat*>(
					model->nodes[node_idx].mesh, 
					0, 
					1, 
					0, 
					{ object_transform.PeekComponents(),object_transform.PeekComponents() + 16 }
				);
			}
		}
	}

	return true;
}
