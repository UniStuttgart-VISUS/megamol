/*
 * CameraParamsVirtualViewOverride.cpp
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Mueller. Alle Rechte vorbehalten.
 */

#include "vislib/CameraParamsVirtualViewOverride.h"


/*
 * CameraParamsVirtualViewOverride::CameraParamsVirtualViewOverride
 */
vislib::graphics::CameraParamsVirtualViewOverride
::CameraParamsVirtualViewOverride(void) : Super() {
}


/*
 *CameraParamsVirtualViewOverride::CameraParamsVirtualViewOverride
 */
vislib::graphics::CameraParamsVirtualViewOverride
::CameraParamsVirtualViewOverride(const SmartPtr<CameraParameters>& params)
        : Super(params), overrideValue(params->VirtualViewSize()) {
    this->indicateValueChange();
}



/*
 * CameraParamsVirtualViewOverride::~CameraParamsVirtualViewOverride
 */
vislib::graphics::CameraParamsVirtualViewOverride
::~CameraParamsVirtualViewOverride(void) {
}


/*
 * vislib::graphics::CameraParamsVirtualViewOverride::SetVirtualViewSize
 */
void vislib::graphics::CameraParamsVirtualViewOverride::SetVirtualViewSize(
        const ImageSpaceDimension& viewSize) {
    ASSERT(!this->paramsBase().IsNull());

    if (math::IsEqual(this->TileRect().GetLeft(), 0.0f)
            && math::IsEqual(this->TileRect().GetBottom(), 0.0f)
            && math::IsEqual(this->TileRect().GetRight(), 
            this->overrideValue.Width())
            && math::IsEqual(this->TileRect().GetTop(), 
            this->overrideValue.Height())) {
        math::Point<ImageSpaceType, 2> origin;
        ImageSpaceRectangle rect(origin, viewSize);
        this->SetTileRect(rect);
    }

    this->overrideValue = viewSize;
    this->indicateValueChange();
}


/*
 * vislib::graphics::CameraParamsVirtualViewOverride::VirtualViewSize
 */
const vislib::graphics::ImageSpaceDimension&
vislib::graphics::CameraParamsVirtualViewOverride::VirtualViewSize(void) const {
    ASSERT(!this->paramsBase().IsNull());
    return this->overrideValue;
}


/*
 * vislib::graphics::CameraParamsVirtualViewOverride::operator =
 */
vislib::graphics::CameraParamsVirtualViewOverride& 
vislib::graphics::CameraParamsVirtualViewOverride::operator =(
        const CameraParamsVirtualViewOverride& rhs) {
    Super::operator =(rhs);
    this->SetVirtualViewSize(rhs.overrideValue);
    return *this;
}


/*
 * vislib::graphics::CameraParamsVirtualViewOverride::operator ==
 */
bool vislib::graphics::CameraParamsVirtualViewOverride::operator ==(
        const CameraParamsVirtualViewOverride& rhs) const {
    return (Super::operator ==(rhs) 
        && (this->overrideValue == rhs.overrideValue));
}


/*
 *  vislib::graphics::CameraParamsEyeOverride::preBaseSet
 */
void vislib::graphics::CameraParamsVirtualViewOverride::preBaseSet(
        const SmartPtr<CameraParameters>& params) {
    // TODO: Should probably do Sebastian's black magic I don't know
}


/*
 *  vislib::graphics::CameraParamsVirtualViewOverride::resetOverride
 */
void vislib::graphics::CameraParamsVirtualViewOverride::resetOverride(void) {
    ASSERT(!this->paramsBase().IsNull());
    this->SetVirtualViewSize(this->paramsBase()->VirtualViewSize());
}
