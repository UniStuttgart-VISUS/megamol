/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "cluster/AbstractClusterView.h"
#include <GL/gl.h>

using namespace megamol::core;


/*
 * cluster::AbstractClusterView::AbstractClusterView
 */
cluster::AbstractClusterView::AbstractClusterView(void) : view::AbstractTileView(),
        ClusterControllerClient() {

    // slot initialized in 'ClusterControllerClient::ctor'
    this->MakeSlotAvailable(&this->registerSlot);

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::~AbstractClusterView
 */
cluster::AbstractClusterView::~AbstractClusterView(void) {

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::ResetView
 */
void cluster::AbstractClusterView::ResetView(void) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DButtonState
 */
void cluster::AbstractClusterView::SetCursor2DButtonState(unsigned int btn, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DPosition
 */
void cluster::AbstractClusterView::SetCursor2DPosition(float x, float y) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetInputModifier
 */
void cluster::AbstractClusterView::SetInputModifier(mmcInputModifier mod, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::renderFallbackView
 */
void cluster::AbstractClusterView::renderFallbackView(void) {
    ::glViewport(0, 0, this->getViewportWidth(), this->getViewportHeight());

    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

}
