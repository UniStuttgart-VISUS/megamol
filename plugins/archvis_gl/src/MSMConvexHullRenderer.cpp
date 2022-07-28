/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "MSMConvexHullRenderer.h"

#include <QuickHull.hpp>

#include "ArchVisCalls.h"
#include "mesh_gl/MeshCalls_gl.h"

megamol::archvis_gl::MSMConvexHullRenderer::MSMConvexHullRenderer()
        : m_MSM_callerSlot("getMSM", "Connects the ..."){
    this->m_MSM_callerSlot.SetCompatibleCall<ScaleModelCallDescription>();
    this->MakeSlotAvailable(&this->m_MSM_callerSlot);
}

megamol::archvis_gl::MSMConvexHullRenderer::~MSMConvexHullRenderer() {}

bool megamol::archvis_gl::MSMConvexHullRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    return false;
}

void megamol::archvis_gl::MSMConvexHullRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<mesh_gl::GPUMaterialCollection>();
    //  material_collection_->addMaterial(
    //      this->instance(), "convexhull", {"archvis_gl/convex_hull.vert.glsl", "archvis_gl/convex_hull.frag.glsl"});
}

bool megamol::archvis_gl::MSMConvexHullRenderer::updateMeshCollection() {
    bool retval = false;

    CallScaleModel* msm_call = this->m_MSM_callerSlot.CallAs<CallScaleModel>();
    if (msm_call != nullptr) {

        if (!(*msm_call)(0)) {
            //TODO throw error
        }

        if (msm_call->hasUpdate()) {
            mesh_collection_->clear();
            retval = true;

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

    } else {
        if (mesh_collection_->getMeshes().size() > 0) {
            mesh_collection_->clear();
            retval = true;
        }
    }

    return retval;
}

void megamol::archvis_gl::MSMConvexHullRenderer::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {
    //TODO
}
