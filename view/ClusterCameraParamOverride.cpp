/*
 * ClusterCameraParamOverride.cpp
 *
 * Copyright (C) 2008 - 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterCameraParamOverride.h"

using namespace megamol::core;


/*
 * view::ClusterCameraParamOverride::ClusterCameraParamOverride
 */
view::ClusterCameraParamOverride::ClusterCameraParamOverride(void)
        : vislib::graphics::CameraParamsOverride(),
        projOverridden(false), tileOverridden(false),
        eye(vislib::graphics::CameraParameters::RIGHT_EYE),
        pj(vislib::graphics::CameraParameters::MONO_PERSPECTIVE),
        tile(0, 0, 1, 1), plane(1, 1) {
    // intentionally empty
}


/*
 * view::ClusterCameraParamOverride::ClusterCameraParamOverride
 */
view::ClusterCameraParamOverride::ClusterCameraParamOverride(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params)
        : vislib::graphics::CameraParamsOverride(params),
        projOverridden(false), tileOverridden(false), eye(params->Eye()),
        pj(params->Projection()), tile(), plane(params->VirtualViewSize()) {
    this->tile = params->TileRect(); // has to be set AFTER plane
    this->indicateValueChange();
}


/*
 * view::ClusterCameraParamOverride::~ClusterCameraParamOverride
 */
view::ClusterCameraParamOverride::~ClusterCameraParamOverride(void) {
    // intentionally empty
}


/*
 * view::ClusterCameraParamOverride::SetOverrides
 */
void view::ClusterCameraParamOverride::SetOverrides(
        const view::CallRenderView& call) {
    this->projOverridden = call.IsProjectionSet();
    this->eye = call.GetEye();
    this->pj = call.GetProjectionType();

    this->tileOverridden = call.IsTileSet();
    this->plane.Set(call.VirtualWidth(), call.VirtualHeight());
    this->tile.Set(call.TileX(), call.TileY(), call.TileX() + call.TileWidth(), call.TileY() + call.TileHeight());

    this->indicateValueChange();
}


/*
 * view::ClusterCameraParamOverride::operator=
 */
view::ClusterCameraParamOverride& view::ClusterCameraParamOverride::operator=(
        const view::ClusterCameraParamOverride& rhs) {
    CameraParamsOverride::operator=(rhs);

    this->projOverridden = rhs.projOverridden;
    this->eye = rhs.eye;
    this->pj = rhs.pj;

    this->tileOverridden = rhs.tileOverridden;
    this->tile = rhs.tile;
    this->plane = rhs.plane;

    this->indicateValueChange();
    return *this;
}


/*
 * view::ClusterCameraParamOverride::operator==
 */
bool view::ClusterCameraParamOverride::operator==(
        const view::ClusterCameraParamOverride& rhs) const {
    return (CameraParamsOverride::operator==(rhs)
        && (this->projOverridden == rhs.projOverridden)
        && (this->eye == rhs.eye)
        && (this->pj == rhs.pj)
        && (this->tileOverridden == rhs.tileOverridden)
        && (this->tile == rhs.tile)
        && (this->plane == rhs.plane));
}


/*
 * view::ClusterCameraParamOverride::preBaseSet
 */
void view::ClusterCameraParamOverride::preBaseSet(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params) {
    // intentionally empty
}


/*
 * view::ClusterCameraParamOverride::resetOverride
 */
void view::ClusterCameraParamOverride::resetOverride(void) {
    ASSERT(!this->paramsBase().IsNull());

    this->projOverridden = false;
    this->eye = this->paramsBase()->Eye();
    this->pj = this->paramsBase()->Projection();

    this->tileOverridden = false;
    this->plane = this->paramsBase()->VirtualViewSize();
    this->tile = this->paramsBase()->TileRect();

    this->indicateValueChange();
}
