#include "ClusterMapRenderer.h"
#include "CallClustering_2.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster_gl;

/*
 * ClusterMapRenderer::ClusterMapRenderer
 */
ClusterMapRenderer::ClusterMapRenderer(void)
        : core_gl::view::Renderer2DModuleGL()
        , clusterDataSlot("clusterData", "Input slot for the cluster data") {
    // Caller Slot
    this->clusterDataSlot.SetCompatibleCall<CallClustering_2Description>();
    this->MakeSlotAvailable(&this->clusterDataSlot);
}

/*
 * ClusterMapRenderer::~ClusterMapRenderer
 */
ClusterMapRenderer::~ClusterMapRenderer(void) {
    this->Release();
}

/*
 * ClusterMapRenderer::OnMouseButton
 */
bool ClusterMapRenderer::OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {
    // TODO
    return true;
}

/*
 * ClusterMapRenderer::OnMouseMove
 */
bool ClusterMapRenderer::OnMouseMove(double x, double y) {
    // TODO
    return true;
}

/*
 * ClusterMapRenderer::create
 */
bool ClusterMapRenderer::create(void) {
    // TODO
    return true;
}

/*
 * ClusterMapRenderer::release
 */
void ClusterMapRenderer::release(void) {
    // TODO
}

/*
 * ClusterMapRenderer::GetExtents
 */
bool ClusterMapRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {
    // TODO
    return false;
}

/*
 * ClusterMapRenderer::Render
 */
bool ClusterMapRenderer::Render(core_gl::view::CallRender2DGL& call) {
    // TODO
    return false;
}
