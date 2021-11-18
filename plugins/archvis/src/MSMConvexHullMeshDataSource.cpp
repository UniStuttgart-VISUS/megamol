#include "MSMConvexHullMeshDataSource.h"

#include <QuickHull.hpp>

#include "mesh/MeshCalls.h"

#include "ArchVisCalls.h"

megamol::archvis::MSMConvexHullDataSource::MSMConvexHullDataSource()
        : m_MSM_callerSlot("getMSM", "Connects the ")
        , m_version(0) {
    this->m_MSM_callerSlot.SetCompatibleCall<ScaleModelCallDescription>();
    this->MakeSlotAvailable(&this->m_MSM_callerSlot);
}

megamol::archvis::MSMConvexHullDataSource::~MSMConvexHullDataSource() {}

bool megamol::archvis::MSMConvexHullDataSource::getDataCallback(core::Call& caller) {
    mesh::CallGPUMeshData* lhs_mesh_call = dynamic_cast<mesh::CallGPUMeshData*>(&caller);
    mesh::CallGPUMeshData* rhs_mesh_call = this->m_mesh_rhs_slot.CallAs<mesh::CallGPUMeshData>();

    if (lhs_mesh_call == nullptr) {
        return false;
    }

    std::vector<std::shared_ptr<mesh::GPUMeshCollection>> gpu_mesh_collection;
    // if there is a mesh connection to the right, pass on the mesh collection
    if (rhs_mesh_call != nullptr) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++m_version;
        }
        gpu_mesh_collection = rhs_mesh_call->getData();
    }
    gpu_mesh_collection.push_back(m_mesh_collection.first);


    CallScaleModel* msm_call = this->m_MSM_callerSlot.CallAs<CallScaleModel>();
    if (msm_call == NULL)
        return false;
    if (!(*msm_call)(0))
        return false;

    if (msm_call->hasUpdate()) {
        ++m_version;

        // TODO create mesh
        quickhull::QuickHull<float> qh;
        std::vector<quickhull::Vector3<float>> point_cloud;

        auto msm = msm_call->getData();

        size_t node_cnt = msm->getNodeCount();
        point_cloud.reserve(node_cnt);

        for (int i = 0; i < node_cnt; ++i) {
            point_cloud.push_back(quickhull::Vector3<float>(msm->accessNodePositions()[i].X(),
                msm->accessNodePositions()[i].Y(), msm->accessNodePositions()[i].Z()));
        }

        auto hull = qh.getConvexHull(point_cloud, true, false);
        auto indexBuffer = hull.getIndexBuffer();
        auto vertexBuffer = hull.getVertexBuffer();

        //TODO stuff like actually creating and adding a mesh
    }

    lhs_mesh_call->setData(gpu_mesh_collection, m_version);

    return true;
}

bool megamol::archvis::MSMConvexHullDataSource::getMetaDataCallback(core::Call& caller) {
    return false;
}
