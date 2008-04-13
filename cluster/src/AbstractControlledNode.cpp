/*
 * AbstractControlledNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractControlledNode.h"

#include "vislib/CameraParamsStore.h"
#include "vislib/clustermessages.h"
#include "vislib/ShallowPoint.h"
#include "vislib/ShallowVector.h"
#include "vislib/Trace.h"



/*
 * vislib::net::cluster::AbstractControlledNode::~AbstractControlledNode
 */
vislib::net::cluster::AbstractControlledNode::~AbstractControlledNode(void) {
    // TODO: Implement
}


/*
 * vislib::net::cluster::AbstractControlledNode::AbstractControlledNode
 */
vislib::net::cluster::AbstractControlledNode::AbstractControlledNode(void) 
        : parameters(NULL) {
    this->parameters = new graphics::CameraParamsStore();
}


/*
 * vislib::net::cluster::AbstractControlledNode::onMessageReceived
 */
void vislib::net::cluster::AbstractControlledNode::onMessageReceived(
        const Socket& src, const UINT msgId, const BYTE *body, 
        const SIZE_T cntBody) {
    TRACE(Trace::LEVEL_VL_INFO, "TODO: cam ctrl received %u\n", msgId);

    if (msgId == MSGID_POSITION) {
        graphics::ShallowSceneSpacePoint3D pt(
            (graphics::SceneSpaceType *)(body));
        this->parameters->SetPosition(pt);

    } else if (msgId == MSGID_UP) {
        graphics::ShallowSceneSpaceVector3D vec(
            (graphics::SceneSpaceType *)(body));
        this->parameters->SetUp(vec);
    }
}


/*
 * vislib::net::cluster::AbstractControlledNode::operator =
 */
vislib::net::cluster::AbstractControlledNode& 
vislib::net::cluster::AbstractControlledNode::operator =(
        const AbstractControlledNode& rhs) {
    if (this != &rhs) {
        AbstractClusterNode::operator =(rhs);
        this->parameters = rhs.parameters;
    }
    return *this;
}
