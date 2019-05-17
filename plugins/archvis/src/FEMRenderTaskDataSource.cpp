#include "FEMRenderTaskDataSource.h"

#include "ng_mesh/GPURenderTaskDataCall.h"
#include "ng_mesh/GPUMaterialDataCall.h"
#include "ng_mesh/GPUMeshDataCall.h"

#include "ng_mesh/GPUMeshCollection.h"

#include "FEMDataCall.h"

megamol::archvis::FEMRenderTaskDataSource::FEMRenderTaskDataSource()
	: m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data")
{
	this->m_fem_callerSlot.SetCompatibleCall<FEMDataCallDescription>();
	this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis::FEMRenderTaskDataSource::~FEMRenderTaskDataSource()
{
}

bool megamol::archvis::FEMRenderTaskDataSource::getDataCallback(core::Call & caller)
{
	ngmesh::GPURenderTaskDataCall* rtc = dynamic_cast<ngmesh::GPURenderTaskDataCall*>(&caller);
	if (rtc == NULL)
		return false;

	ngmesh::GPUMaterialDataCall* mtlc = this->m_material_callerSlot.CallAs<ngmesh::GPUMaterialDataCall>();
	if (mtlc == NULL)
		return false;

	if (!(*mtlc)(0))
		return false;

	ngmesh::GPUMeshDataCall* mc = this->m_mesh_callerSlot.CallAs<ngmesh::GPUMeshDataCall>();
	if (mc == NULL)
		return false;

	if (!(*mc)(0))
		return false;

	auto gpu_mtl_storage = mtlc->getMaterialStorage();
	auto gpu_mesh_storage = mc->getGPUMeshes();

	FEMDataCall* fem_call = this->m_fem_callerSlot.CallAs<FEMDataCall>();
	if (fem_call == NULL)
		return false;

	if (!(*fem_call)(0))
		return false;

	if (fem_call->getUpdateFlag())
	{
        m_gpu_render_tasks->clear();

		for (auto& sub_mesh : gpu_mesh_storage->getSubMeshData())
		{
			auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
			auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;
		
			std::vector<ngmesh::DrawElementsCommand> draw_commands(1, sub_mesh.sub_mesh_draw_command);
		
			std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR >> object_transform(1000);
			typedef std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR >> PerTaskData;
		
			GLfloat scale = 1.0f;
			object_transform[0].SetAt(0, 0, scale);
			object_transform[0].SetAt(1, 1, scale);
			object_transform[0].SetAt(2, 2, scale);
		
			object_transform[0].SetAt(3, 3, 1.0f);
		
			object_transform[9].SetAt(0, 3, 0.0f);
			object_transform[9].SetAt(1, 3, 0.0f);
			object_transform[9].SetAt(2, 3, 0.0f);
		
			m_gpu_render_tasks->addRenderTasks(shader, gpu_batch_mesh, draw_commands, object_transform);
		}

        auto const& node_deformation = fem_call->getFEMData()->getNodeDeformations();

        m_gpu_render_tasks->addPerFrameDataBuffer(node_deformation, 1);

        // TODO get transfer function texture and add as per frame data
        std::vector<GLuint64> texture_handles;
        auto textures = gpu_mtl_storage->getMaterials().front().textures_names;
        for (auto texture : textures)
        {
            texture_handles.push_back(glGetTextureHandleARB(texture));
            glMakeTextureHandleResidentARB(texture_handles.back());
        }
        m_gpu_render_tasks->addPerFrameDataBuffer(texture_handles, 2);

        fem_call->clearUpdateFlag();
    } else {
        // TODO get transfer function texture and add as per frame data
        std::vector<GLuint64> texture_handles;
        auto textures = gpu_mtl_storage->getMaterials().front().textures_names;
        for (auto texture : textures) {
            texture_handles.push_back(glGetTextureHandleARB(texture));
            glMakeTextureHandleResidentARB(texture_handles.back());
        }
        m_gpu_render_tasks->updatePerFrameDataBuffer(texture_handles, 2);
    }

	rtc->setRenderTaskData(m_gpu_render_tasks.get());

	return true;
}
