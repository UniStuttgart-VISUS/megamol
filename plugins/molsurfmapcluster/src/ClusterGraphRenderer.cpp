#include "ClusterGraphRenderer.h"
#include "CallClustering_2.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster;

/*
 * ClusterGraphRenderer::ClusterGraphRenderer
 */
ClusterGraphRenderer::ClusterGraphRenderer(void)
    : view::Renderer2DModule(), clusterDataSlot("clusterData", "Input slot for the cluster data") {
    // Caller Slot
    this->clusterDataSlot.SetCompatibleCall<CallClustering_2Description>();
    this->MakeSlotAvailable(&this->clusterDataSlot);
}

/*
 * ClusterGraphRenderer::~ClusterGraphRenderer
 */
ClusterGraphRenderer::~ClusterGraphRenderer(void) { this->Release(); }

/*
 * ClusterGraphRenderer::OnMouseButton
 */
bool ClusterGraphRenderer::OnMouseButton(
    view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {
    // TODO
    return true;
}

/*
 * ClusterGraphRenderer::OnMouseMove
 */
bool ClusterGraphRenderer::OnMouseMove(double x, double y) {
    // TODO
    return true;
}

/*
 * ClusterGraphRenderer::create
 */
bool ClusterGraphRenderer::create(void) {
    // TODO
    return true;
}

/*
 * ClusterGraphRenderer::release
 */
void ClusterGraphRenderer::release(void) {
    // TODO
}

/*
 * ClusterGraphRenderer::GetExtents
 */
bool ClusterGraphRenderer::GetExtents(view::CallRender2D& call) {
    // TODO
    return false;
}

/*
 * ClusterGraphRenderer::Render
 */
bool ClusterGraphRenderer::Render(view::CallRender2D& call) {
    // TODO
    return false;
}
