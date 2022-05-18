/*
 * Renderer3DModule.cpp
 *
 * Copyright (C) 2018, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRender3D.h"
#include "stdafx.h"


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
        // calculate union of bounding and clip boxes, respectively
        // if only the chained bounding (or clip) box is valid, use it directly instead of the union
        if (call.GetBoundingBoxes().IsBoundingBoxValid() && chainedCall->GetBoundingBoxes().IsBoundingBoxValid()) {
            auto mybb = call.GetBoundingBoxes().BoundingBox();
            mybb.Union(chainedCall->GetBoundingBoxes().BoundingBox());
            call.AccessBoundingBoxes().SetBoundingBox(mybb);
        } else if (chainedCall->GetBoundingBoxes().IsBoundingBoxValid()) {
            auto mybb = chainedCall->GetBoundingBoxes().BoundingBox();
            call.AccessBoundingBoxes().SetBoundingBox(mybb);
        }

        if (call.GetBoundingBoxes().IsClipBoxValid() && chainedCall->GetBoundingBoxes().IsClipBoxValid()) {
            auto mycb = call.GetBoundingBoxes().ClipBox();
            mycb.Union(chainedCall->GetBoundingBoxes().ClipBox());
            call.AccessBoundingBoxes().SetClipBox(mycb);
        } else if (chainedCall->GetBoundingBoxes().IsClipBoxValid()) {
            auto mycb = chainedCall->GetBoundingBoxes().ClipBox();
            call.AccessBoundingBoxes().SetClipBox(mycb);
        }

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
