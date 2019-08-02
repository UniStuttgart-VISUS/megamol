/*
 * CameraParamsTileRectOverride.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/graphics/CameraParamsTileRectOverride.h"
#include "vislib/assert.h"
#include "vislib/math/mathfunctions.h"


/*
 * vislib::graphics::CameraParamsTileRectOverride::CameraParamsTileRectOverride
 */
vislib::graphics::CameraParamsTileRectOverride::CameraParamsTileRectOverride(void)
    : CameraParamsOverride(), fullSize(true), tileRect() {}


/*
 * vislib::graphics::CameraParamsTileRectOverride::CameraParamsTileRectOverride
 */
vislib::graphics::CameraParamsTileRectOverride::CameraParamsTileRectOverride(
    const vislib::SmartPtr<vislib::graphics::CameraParameters>& params)
    : CameraParamsOverride(params), fullSize(true), tileRect(params->TileRect()) {
    this->fullSize =
        (math::IsEqual(this->tileRect.GetLeft(), 0.0f) && math::IsEqual(this->tileRect.GetBottom(), 0.0f) &&
            math::IsEqual(this->tileRect.GetRight(), params->VirtualViewSize().Width()) &&
            math::IsEqual(this->tileRect.GetTop(), params->VirtualViewSize().Height()));
    this->indicateValueChange();
}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::~CameraParamsTileRectOverride
 */
vislib::graphics::CameraParamsTileRectOverride::~CameraParamsTileRectOverride(void) {}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::ResetTileRect
 */
void vislib::graphics::CameraParamsTileRectOverride::ResetTileRect(void) {
    ASSERT(!this->paramsBase().IsNull());
    assign_and_sync(this->fullSize, true);
    if (this->TileRect().Width() != this->paramsBase()->VirtualViewSize().Width() ||
        this->TileRect().Height() != this->paramsBase()->VirtualViewSize().Height()) {
        this->tileRect.SetNull();
        this->tileRect.SetSize(this->paramsBase()->VirtualViewSize());
        this->indicateValueChange();
    }
}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::SetTileRect
 */
void vislib::graphics::CameraParamsTileRectOverride::SetTileRect(
    const vislib::math::Rectangle<vislib::graphics::ImageSpaceType>& tileRect) {
    ASSERT(!this->paramsBase().IsNull());
    assign_and_sync(this->tileRect, tileRect);
    assign_and_sync(this->fullSize,
        (math::IsEqual(this->tileRect.GetLeft(), 0.0f) && math::IsEqual(this->tileRect.GetBottom(), 0.0f) &&
            math::IsEqual(this->tileRect.GetRight(), this->paramsBase()->VirtualViewSize().Width()) &&
            math::IsEqual(this->tileRect.GetTop(), this->paramsBase()->VirtualViewSize().Height())));
}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::TileRect
 */
const vislib::math::Rectangle<vislib::graphics::ImageSpaceType>&
vislib::graphics::CameraParamsTileRectOverride::TileRect(void) const {
    ASSERT(!this->paramsBase().IsNull());
    if (this->fullSize) {
        if (this->tileRect.Width() != this->paramsBase()->VirtualViewSize().Width() ||
            this->tileRect.Height() != this->paramsBase()->VirtualViewSize().Height()) {
            this->tileRect.SetNull();
            this->tileRect.SetSize(this->paramsBase()->VirtualViewSize());
            this->indicateValueChange_Const();
        }
    }
    return this->tileRect;
}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::operator=
 */
vislib::graphics::CameraParamsTileRectOverride& vislib::graphics::CameraParamsTileRectOverride::operator=(
    const vislib::graphics::CameraParamsTileRectOverride& rhs) {
    CameraParamsOverride::operator=(rhs);
    this->tileRect = rhs.tileRect;
    this->fullSize = rhs.fullSize;
    this->indicateValueChange();
    return *this;
}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::operator==
 */
bool vislib::graphics::CameraParamsTileRectOverride::operator==(
    const vislib::graphics::CameraParamsTileRectOverride& rhs) const {
    return (
        CameraParamsOverride::operator==(rhs) && (this->tileRect == rhs.tileRect) && (this->fullSize == rhs.fullSize));
}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::preBaseSet
 */
void vislib::graphics::CameraParamsTileRectOverride::preBaseSet(const SmartPtr<CameraParameters>& params) {
    if (params.IsNull()) {
        return;
    }

    if (this->fullSize) {
        if (this->tileRect.Width() != params->VirtualViewSize().Width() ||
            this->tileRect.Height() != params->VirtualViewSize().Height()) {
            this->tileRect.SetNull();
            this->tileRect.SetSize(params->VirtualViewSize());
            this->indicateValueChange();
        }

    } else if (!this->paramsBase().IsNull()) {
        // scale the tile from the old view size to the new view size and hope
        // that the caller will set a tile making more sense.
        ImageSpaceType const scaleX = params->VirtualViewSize().Width() / this->paramsBase()->VirtualViewSize().Width();
        ImageSpaceType const scaleY =
            params->VirtualViewSize().Height() / this->paramsBase()->VirtualViewSize().Height();
        if (!vislib::math::almost_equal(scaleX, static_cast<ImageSpaceType>(1.0f)) ||
            !vislib::math::almost_equal(scaleY, static_cast<ImageSpaceType>(1.0f))) {
            this->tileRect.Set(this->tileRect.Left() * scaleX, this->tileRect.Bottom() * scaleY,
                this->tileRect.Right() * scaleX, this->tileRect.Top() * scaleY);
            this->indicateValueChange();
        }
    }
}


/*
 *  vislib::graphics::CameraParamsTileRectOverride::resetOverride
 */
void vislib::graphics::CameraParamsTileRectOverride::resetOverride(void) {
    ASSERT(!this->paramsBase().IsNull());
    assign_and_sync(this->tileRect, this->paramsBase()->TileRect());
    assign_and_sync(this->fullSize,
        (math::IsEqual(this->tileRect.GetLeft(), 0.0f) && math::IsEqual(this->tileRect.GetBottom(), 0.0f) &&
            math::IsEqual(this->tileRect.GetRight(), this->paramsBase()->VirtualViewSize().Width()) &&
            math::IsEqual(this->tileRect.GetTop(), this->paramsBase()->VirtualViewSize().Height())));
}
