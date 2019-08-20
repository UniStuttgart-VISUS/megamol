#include "stdafx.h"

#include "glTFRenderTasksDataSource.h"
#include "tiny_gltf.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "mesh/CallGPURenderTaskData.h"
#include "mesh/CallGPUMaterialData.h"
#include "mesh/CallGPUMeshData.h"

megamol::mesh::GlTFRenderTasksDataSource::GlTFRenderTasksDataSource()
	: m_glTF_callerSlot("getGlTFFile", "Connects the data source with a loaded glTF file")
{
	this->m_glTF_callerSlot.SetCompatibleCall<CallGlTFDataDescription>();
	this->MakeSlotAvailable(&this->m_glTF_callerSlot);
}

megamol::mesh::GlTFRenderTasksDataSource::~GlTFRenderTasksDataSource()
{
}

bool megamol::mesh::GlTFRenderTasksDataSource::getDataCallback(core::Call & caller)
{
	CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL)
		return false;

    std::shared_ptr<GPURenderTaskCollection> rt_collection(nullptr);
    
    if (lhs_rtc->getRenderTaskData() == nullptr){
        rt_collection = this->m_gpu_render_tasks;
        lhs_rtc->setRenderTaskData(rt_collection);
    } else {
        rt_collection = lhs_rtc->getRenderTaskData();
    }

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

	CallGlTFData* gltf_call = this->m_glTF_callerSlot.CallAs<CallGlTFData>();
	if (gltf_call == NULL)
		return false;

	if (!(*gltf_call)(0))
		return false;

	auto gpu_mtl_storage = mtlc->getMaterialStorage();
	auto gpu_mesh_storage = mc->getGPUMeshes();

	//TODO nullptr check

	if (gltf_call->getUpdateFlag())
	{
		//rt_collection->clear();
        if (!m_rt_collection_indices.empty())
        {
            // TODO delete all exisiting render task from this module
            for (auto& rt_idx : m_rt_collection_indices)
            {
                rt_collection->deleteSingleRenderTask(rt_idx);   
            }

            m_rt_collection_indices.clear();
        }

		auto model = gltf_call->getGlTFModel();

		for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++)
		{
			if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1)
			{
				vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;

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
						object_transform.SetAt(0, 3, translation[0]);
						object_transform.SetAt(1, 3, translation[1]);
						object_transform.SetAt(2, 3, translation[2]);
					}

					if (scale.size() != 0) {

					}

					if (rotation.size() != 0) {

					}
				}

				// compute submesh offset by iterating over all meshes before the given mesh and summing up their primitive counts
				size_t submesh_offset = 0;
				for (int mesh_idx = 0; mesh_idx < model->nodes[node_idx].mesh; ++mesh_idx)
				{
					submesh_offset += model->meshes[mesh_idx].primitives.size();
				}

				auto primitive_cnt = model->meshes[model->nodes[node_idx].mesh].primitives.size();
				for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx)
				{
					//auto const& sub_mesh = gpu_mesh_storage->getSubMeshData()[model->nodes[node_idx].mesh];
					auto const& sub_mesh = gpu_mesh_storage->getSubMeshData()[submesh_offset + primitive_idx];
					auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
					auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

					size_t rt_idx = rt_collection->addSingleRenderTask(
						shader,
						gpu_batch_mesh,
						sub_mesh.sub_mesh_draw_command,
						object_transform);

                    m_rt_collection_indices.push_back(rt_idx);
				}
			}
		}

		// add some lights to the scene to test the per frame buffers
        struct LightParams {
            float x,y,z,intensity;
		};

		// Place lights in icosahedron pattern
        float x = 0.525731112119133606f * 1000.0f;
        float z = 0.850650808352039932f * 1000.0f;

        std::vector<LightParams> lights = {
			{-x, 0.0f, z,  1.0f},
			{x, 0.0f, z,   1.0f}, 
			{-x, 0.0f, -z, 1.0f},
			{x, 0.0f, -z,  1.0f},
			{0.0f, z, x,   1.0f},
			{0.0f, z, -x,  1.0f},
			{0.0f, -z, x,  1.0f},
			{0.0f, -z, -x, 1.0f},
			{z, x, 0.0f,   1.0f},
			{-z, x, 0.0f,  1.0f},
			{z, -x, 0.0f,  1.0f},
			{-z, -x, 0.0f, 1.0f}
        };

		// Add a key light
		lights.push_back({-5000.0,5000.0,-5000.0,1000.0f});

		rt_collection->addPerFrameDataBuffer(lights,1);

		gltf_call->clearUpdateFlag();
	}


    CallGPURenderTaskData* rhs_rtc = this->m_renderTask_callerSlot.CallAs<CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setRenderTaskData(rt_collection);

        (*rhs_rtc)(0);
    }

	return true;
}
