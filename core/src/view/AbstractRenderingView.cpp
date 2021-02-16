/*
 * AbstractRenderingView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/AbstractRenderingView.h"
#include "mmcore/AbstractNamedObject.h"
#include "vislib/String.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/sysfunctions.h"
#include "mmcore/utility/sys/Thread.h"

using namespace megamol::core;


/*
 * view::AbstractRenderingView::AbstractTitleRenderer::AbstractTitleRenderer
 */
view::AbstractRenderingView::AbstractTitleRenderer::AbstractTitleRenderer(void) {
    // intentionally empty
}


/*
 * view::AbstractRenderingView::AbstractTitleRenderer::~AbstractTitleRenderer
 */
view::AbstractRenderingView::AbstractTitleRenderer::~AbstractTitleRenderer(void) {
    // intentionally empty
}

/*
 * view::AbstractRenderingView::AbstractRenderingView
 */
view::AbstractRenderingView::AbstractRenderingView(void) : AbstractView(),
        overrideBkgndCol(NULL),
        bkgndColSlot("backCol", "The views background colour") {

    this->bkgndCol[0] = 0.0f;
    this->bkgndCol[1] = 0.0f;
    this->bkgndCol[2] = 0.125f;

    this->bkgndColSlot << new param::ColorParam(this->bkgndCol[0], this->bkgndCol[1], this->bkgndCol[2], 1.0f);
    this->MakeSlotAvailable(&this->bkgndColSlot);
}


/*
 * view::AbstractRenderingView::~AbstractRenderingView
 */
view::AbstractRenderingView::~AbstractRenderingView(void) {
    this->overrideBkgndCol = glm::vec4(0,0,0,0); // DO NOT DELETE
}


/*
* view::AbstractRenderingView::bkgndColour
*/
glm::vec4 view::AbstractRenderingView::BkgndColour(void) const {
    if (this->bkgndColSlot.IsDirty()) {
        this->bkgndColSlot.ResetDirty();
        this->bkgndColSlot.Param<param::ColorParam>()->Value(this->bkgndCol[0], this->bkgndCol[1], this->bkgndCol[2]);
    }
    return this->bkgndCol;
}
