/*
 * AbstractRendererDeferred3D.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractRendererDeferred3D.h"
#include "mmcore/view/CallRenderDeferred3D.h"
#include "mmcore/view/Renderer3DModule.h"


using namespace megamol::core;


/*
 * view::AbstractRendererDeferred3D::AbstractRendererDeferred3D
 */
view::AbstractRendererDeferred3D::AbstractRendererDeferred3D(void) : Renderer3DModule(),
        rendererSlot("renderingDS", "Connects the deferred renderer to another renderer") {

    this->rendererSlot.SetCompatibleCall<CallRenderDeferred3DDescription>();
    this->MakeSlotAvailable(&this->rendererSlot);
}


/*
 * view::AbstractRendererDeferred3D::~AbstractRendererDeferred3D
 */
view::AbstractRendererDeferred3D::~AbstractRendererDeferred3D(void) {
    // intentionally empty
}
