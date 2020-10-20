#include "stdafx.h"

#include "glTFRenderTasksDataSource.h"
#include "tiny_gltf.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "mesh/MeshCalls.h"

megamol::mesh::GlTFRenderTasksDataSource::GlTFRenderTasksDataSource()
	: m_glTF_callerSlot("gltfModels", "Connects a collection of loaded glTF files")
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

    syncRenderTaskCollection(lhs_rtc);

	CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<CallGPUMaterialData>();
	if (mtlc == NULL)
		return false;

	if (!(*mtlc)(0))
		return false;

	CallGPUMeshData* mc = this->m_mesh_slot.CallAs<CallGPUMeshData>();
	if (mc == NULL)
		return false;

	if (!(*mc)(0))
		return false;

	CallGlTFData* gltf_call = this->m_glTF_callerSlot.CallAs<CallGlTFData>();
	if (gltf_call == NULL)
		return false;

	if (!(*gltf_call)(0))
		return false;

	auto gpu_mtl_storage = mtlc->getData();
	auto gpu_mesh_storage = mc->getData();

	//TODO nullptr check

	if (gltf_call->hasUpdate())
    {
        ++m_version;

		for (auto& identifier : m_rendertask_collection.second) {
            m_rendertask_collection.first->deleteRenderTask(identifier);
		}
        m_rendertask_collection.second.clear();

		auto model = gltf_call->getData().second;

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

				auto primitive_cnt = model->meshes[model->nodes[node_idx].mesh].primitives.size();
				for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx)
				{
                    std::string sub_mesh_identifier = gltf_call->getData().first +
                                                      model->meshes[model->nodes[node_idx].mesh].name + "_" +
                                                      std::to_string(primitive_idx);
                    auto const& sub_mesh = gpu_mesh_storage->getSubMesh(sub_mesh_identifier);
                    //TODO check if submesh existed...
                    auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;
                    auto const& shader = gpu_mtl_storage->getMaterials().begin()->second.shader_program;

					std::string rt_identifier(std::string(this->FullName()) + "_" + sub_mesh_identifier);
                    m_rendertask_collection.first->addRenderTask(rt_identifier, shader, gpu_batch_mesh,
                        sub_mesh.sub_mesh_draw_command, object_transform);
                    m_rendertask_collection.second.push_back(rt_identifier);
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

		m_rendertask_collection.first->deletePerFrameDataBuffer(1);
		m_rendertask_collection.first->addPerFrameDataBuffer("lights",lights,1);
    }

    if (lhs_rtc->version() < m_version) {
        lhs_rtc->setData(m_rendertask_collection.first, m_version);
    }


    CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(m_rendertask_collection.first,0);

        (*rhs_rtc)(0);
    }

	return true;
}

bool megamol::mesh::GlTFRenderTasksDataSource::getMetaDataCallback(core::Call& caller) {

    AbstractGPURenderTaskDataSource::getMetaDataCallback(caller);

    auto gltf_call = m_glTF_callerSlot.CallAs<CallGlTFData>();

    if (gltf_call == nullptr) {
        return false;
    }

    if (!(*gltf_call)(1)) {
        return false;
    }

    return true;
}
