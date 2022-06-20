/*
 * Renderer3DModule.cpp
 *
 * Copyright (C) 2018, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRender3D.h"


namespace megamol::core::view {
/*
 * Renderer3DModule::Renderer3DModule
 */
Renderer3DModule::Renderer3DModule(void) : RendererModule<CallRender3D, Module>() {
    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->chainRenderSlot);

    // Callback should already be set by RendererModule
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * Renderer3DModule::~Renderer3DModule
 */
Renderer3DModule::~Renderer3DModule(void) {
    // intentionally empty
}

/*
 * Renderer3DModule::GetExtentsChain
 */
bool Renderer3DModule::GetExtentsChain(CallRender3D& call) {
    CallRender3D* chainedCall = this->chainRenderSlot.CallAs<CallRender3D>();
    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // set bounding boxes to invalid before calling to the right, in case the chained renderer
        // does not provide them at all. this would result in the possibly outdated bounding box
        // from the left hand side being used
        chainedCall->AccessBoundingBoxes().Clear();

        // chain through the get extents call
        (*chainedCall)(view::AbstractCallRender::FnGetExtents);
    }

    // TODO extents magic


    // get our own extents
    // set bounding boxes to invalid before getting own extent, in case the function
    // does not provide them at all. this would result in the possibly outdated bounding box
    // from the left hand side being used
    call.AccessBoundingBoxes().Clear();
    this->GetExtents(call);

    if (chainedCall != nullptr) {
        const auto& mybb = call.AccessBoundingBoxes().BoundingBox();
        const auto& otherbb = chainedCall->AccessBoundingBoxes().BoundingBox();
        const auto& mycb = call.AccessBoundingBoxes().ClipBox();
        const auto& othercb = chainedCall->AccessBoundingBoxes().ClipBox();

        auto newbb = mybb;
        auto newcb = mycb;

        if (call.GetBoundingBoxes().IsBoundingBoxValid() && chainedCall->GetBoundingBoxes().IsBoundingBoxValid()) {
            newbb.Union(otherbb);
        } else if (chainedCall->GetBoundingBoxes().IsBoundingBoxValid()) {
            newbb = otherbb; // just override for the call
        }                    // we ignore the other two cases as they both lead to usage of the already set mybb

        if (call.GetBoundingBoxes().IsClipBoxValid() && chainedCall->GetBoundingBoxes().IsClipBoxValid()) {
            newcb.Union(othercb);
        } else if (chainedCall->GetBoundingBoxes().IsClipBoxValid()) {
            newcb = othercb; // just override for the call
        }                    // we ignore the other two cases as they both lead to usage of the already set mycb

        call.AccessBoundingBoxes().SetBoundingBox(newbb);
        call.AccessBoundingBoxes().SetClipBox(newcb);

        // TODO machs richtig
        call.SetTimeFramesCount(chainedCall->TimeFramesCount());
    }

    return true;
}

/*
 * Renderer3DModule::RenderChain
 */
bool Renderer3DModule::RenderChain(CallRender3D& call) {
    auto leftSlotParent = call.PeekCallerSlot()->Parent();
    std::shared_ptr<const view::AbstractView> viewptr =
        std::dynamic_pointer_cast<const view::AbstractView>(leftSlotParent);

    this->PreRender(call);

    CallRender3D* chainedCall = this->chainRenderSlot.CallAs<CallRender3D>();

    if (chainedCall != nullptr) {
        // copy the incoming call to the output
        *chainedCall = call;

        // chain through the render call
        (*chainedCall)(view::AbstractCallRender::FnRender);

        call = *chainedCall;
    }

    // render our own stuff
    this->Render(call);

    return true;
}

/*
 * Renderer3DModule::PreRender
 */
void Renderer3DModule::PreRender(CallRender3D& call) {
    // intentionally empty
}
} // namespace megamol::core::view
