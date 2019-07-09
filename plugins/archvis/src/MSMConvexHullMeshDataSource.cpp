#include "MSMConvexHullMeshDataSource.h"

#include "MSMDataCall.h"
#include "mesh/CallGPUMeshData.h"

#include <QuickHull.hpp>

megamol::archvis::MSMConvexHullDataSource::MSMConvexHullDataSource() 
: m_MSM_callerSlot("getMSM", "Connects the "), m_MSM_hash(0) {
    this->m_MSM_callerSlot.SetCompatibleCall<MSMDataCallDescription>();
    this->MakeSlotAvailable(&this->m_MSM_callerSlot);
}

megamol::archvis::MSMConvexHullDataSource::~MSMConvexHullDataSource() {}

bool megamol::archvis::MSMConvexHullDataSource::create() { 
    m_gpu_meshes = std::make_shared<mesh::GPUMeshCollection>();

    m_bbox = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    return true;
}

bool megamol::archvis::MSMConvexHullDataSource::getDataCallback(core::Call& caller) { 
    mesh::CallGPUMeshData* mc = dynamic_cast<mesh::CallGPUMeshData*>(&caller);
    if (mc == NULL) return false;

    MSMDataCall* msm_call = this->m_MSM_callerSlot.CallAs<MSMDataCall>();
    if (msm_call == NULL) return false;

    if (!(*msm_call)(0)) return false;

    if (this->m_MSM_hash == msm_call->DataHash()) {
        return true;
    }

    //TODO create mesh
    quickhull::QuickHull<float> qh;
    std::vector<quickhull::Vector3<float>> point_cloud;

    auto msm = msm_call->getMSM();

    size_t node_cnt = msm->getNodeCount();
    point_cloud.reserve(node_cnt);

    for (int i = 0; i < node_cnt; ++i)
    {
        point_cloud.push_back(quickhull::Vector3<float>(
            msm->accessNodePositions()[i].X(),
            msm->accessNodePositions()[i].Y(),
            msm->accessNodePositions()[i].Z())
        );
    }

    auto hull = qh.getConvexHull(point_cloud, true, false);
    auto indexBuffer = hull.getIndexBuffer();
    auto vertexBuffer = hull.getVertexBuffer();

    mc->setGPUMeshes(m_gpu_meshes);

    this->m_MSM_hash = msm_call->DataHash();

    return true;
}
