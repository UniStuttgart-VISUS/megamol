#include "ClusterGraphRenderer.h"
#include "CallClustering_2.h"

#include "mmcore/param/IntParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster;

/*
 * ClusterGraphRenderer::ClusterGraphRenderer
 */
ClusterGraphRenderer::ClusterGraphRenderer(void)
    : core_gl::view::Renderer2DModuleGL()
    , clusterDataSlot("clusterData", "Input slot for the cluster data")
    , viewportHeightParam("viewportHeight", "Height of the viewport")
    , viewportWidthParam("viewportWidth", "Width of the viewport") {
    // Caller Slot
    this->clusterDataSlot.SetCompatibleCall<CallClustering_2Description>();
    this->MakeSlotAvailable(&this->clusterDataSlot);

    // Parameter Slots
    this->viewportHeightParam.SetParameter(new param::IntParam(1440, 100, 10800));
    this->MakeSlotAvailable(&this->viewportHeightParam);

    this->viewportWidthParam.SetParameter(new param::IntParam(2560, 100, 10800));
    this->MakeSlotAvailable(&this->viewportWidthParam);
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
    return false;
}

/*
 * ClusterGraphRenderer::OnMouseMove
 */
bool ClusterGraphRenderer::OnMouseMove(double x, double y) {
    // TODO
    return false;
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
bool ClusterGraphRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {
    call.AccessBoundingBoxes().SetBoundingBox(0, 0, static_cast<float>(this->viewportWidthParam.Param<param::IntParam>()->Value()),
        static_cast<float>(this->viewportHeightParam.Param<param::IntParam>()->Value()));

    CallClustering_2* cc = this->clusterDataSlot.CallAs<CallClustering_2>();
    if (cc == nullptr) return false;

    if (!(*cc)(CallClustering_2::CallForGetExtent)) return false;

    return true;
}

/*
 * ClusterGraphRenderer::Render
 */
bool ClusterGraphRenderer::Render(core_gl::view::CallRender2DGL& call) {
    CallClustering_2* cc = this->clusterDataSlot.CallAs<CallClustering_2>();
    if (cc == nullptr) return false;

    if (!(*cc)(CallClustering_2::CallForGetData)) return false;

    return true;
}
