/*
 * AbstractRenderingView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractRenderingView.h"
#include "param/BoolParam.h"
#include "param/StringParam.h"
#include "utility/ColourParser.h"

using namespace megamol::core;


/*
 * view::AbstractRenderingView::AbstractRenderingView
 */
view::AbstractRenderingView::AbstractRenderingView(void) : AbstractView(),
        overrideBkgndCol(NULL), overrideViewport(NULL),
        bkgndColSlot("backCol", "The views background colour"),
        softCursor(false), softCursorSlot("softCursor", "Bool flag to activate software cursor rendering") {

    this->bkgndCol[0] = 0.0f;
    this->bkgndCol[1] = 0.0f;
    this->bkgndCol[2] = 0.125f;

    this->bkgndColSlot << new param::StringParam(utility::ColourParser::ToString(
        this->bkgndCol[0], this->bkgndCol[1], this->bkgndCol[2]));
    this->MakeSlotAvailable(&this->bkgndColSlot);

    this->softCursorSlot << new param::BoolParam(this->softCursor);
    this->MakeSlotAvailable(&this->softCursorSlot);

}


/*
 * view::AbstractRenderingView::~AbstractRenderingView
 */
view::AbstractRenderingView::~AbstractRenderingView(void) {
    this->overrideBkgndCol = NULL; // DO NOT DELETE
    this->overrideViewport = NULL; // DO NOT DELETE
}


/*
 * view::AbstractRenderingView::bkgndColour
 */
const float *view::AbstractRenderingView::bkgndColour(void) const {
    if (this->bkgndColSlot.IsDirty()) {
        this->bkgndColSlot.ResetDirty();
        utility::ColourParser::FromString(
            this->bkgndColSlot.Param<param::StringParam>()->Value(),
            this->bkgndCol[0], this->bkgndCol[1], this->bkgndCol[2]);
    }
    return this->bkgndCol;
}


/*
 * view::AbstractRenderingView::showSoftCursor
 */
bool view::AbstractRenderingView::showSoftCursor(void) const {
    if (this->softCursorSlot.IsDirty()) {
        this->softCursorSlot.ResetDirty();
        this->softCursor = this->softCursorSlot.Param<param::BoolParam>()->Value();
    }
    return this->softCursor;
}
