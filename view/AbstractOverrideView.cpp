/*
 * AbstractOverrideView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractOverrideView.h"

using namespace megamol::core;


/*
 * view::AbstractOverrideView::AbstractOverrideView
 */
view::AbstractOverrideView::AbstractOverrideView(void) : AbstractView(),
        renderViewSlot("renderView", "Slot for outgoing rendering requests to other views") {

    this->renderViewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->renderViewSlot);

}


/*
 * view::AbstractOverrideView::~AbstractOverrideView
 */
view::AbstractOverrideView::~AbstractOverrideView(void) {

    // TODO: Implement

}


/*
 * view::AbstractOverrideView::packMouseCoordinates
 */
void view::AbstractOverrideView::packMouseCoordinates(float &x, float &y) {
    // intentionally empty
    // do something smart in the derived classes
}
