#include "stdafx.h"

#include "glTFRenderTasksDataSource.h"
#include "tiny_gltf.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "ng_mesh/GPURenderTaskDataCall.h"
#include "ng_mesh/GPUMaterialDataCall.h"
#include "ng_mesh/GPUMeshDataCall.h"
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
	GPURenderTaskDataCall* rtc = dynamic_cast<GPURenderTaskDataCall*>(&caller);
	if (rtc == NULL)
		return false;

	GPUMaterialDataCall* mtlc = this->m_material_callerSlot.CallAs<GPUMaterialDataCall>();
	if (mtlc == NULL)
		return false;

	if (!(*mtlc)(0))
		return false;

	GPUMeshDataCall* mc = this->m_mesh_callerSlot.CallAs<GPUMeshDataCall>();
	if (mc == NULL)
		return false;

	if (!(*mc)(0))
		return false;

	GlTFDataCall* gltf_call = this->m_glTF_callerSlot.CallAs<GlTFDataCall>();
	if (gltf_call == NULL)
		return false;

	if (!(*gltf_call)(0))
		return false;

	auto gpu_mtl_storage = mtlc->getMaterialStorage();
	auto gpu_mesh_storage = mc->getGPUMeshes();

	//TODO nullptr check

	if (gltf_call->getUpdateFlag())
	{
		auto model = gltf_call->getGlTFModel();

		for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++)
		{
			if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1)
			{
				std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> object_transform(1);

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
						object_transform[0].SetAt(0, 3, translation[0]);
						object_transform[0].SetAt(1, 3, translation[1]);
						object_transform[0].SetAt(2, 3, translation[2]);
					}

					if (scale.size() != 0) {

					}

					if (rotation.size() != 0) {

					}
				}

				//TODO bounding box ?

				auto const& sub_mesh = gpu_mesh_storage->getSubMeshData()[model->nodes[node_idx].mesh];
				auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
				auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

				m_gpu_render_tasks->addSingleRenderTask(
					shader,
					gpu_batch_mesh,
					sub_mesh.sub_mesh_draw_command,
					object_transform);
			}
		}

		gltf_call->clearUpdateFlag();
	}

	rtc->setRenderTaskData(m_gpu_render_tasks.get());

	return true;
}
