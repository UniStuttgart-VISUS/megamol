/*
 * AbstractQuartzRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "AbstractQuartzRenderer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/light/CallLight.h"
#include "stdafx.h"

namespace megamol {
namespace demos_gl {

/*
 * AbstractQuartzRenderer::AbstractQuartzRenderer
 */
AbstractQuartzRenderer::AbstractQuartzRenderer(void)
        : AbstractQuartzModule()
        , clipPlaneSlot("clipplane", "Slot connecting to the clipping plane provider")
        , lightsSlot("lights", "Lights are retrieved over this slot.")
        , showClipPlanePolySlot("showClipPlanePoly", "Shows/Hides the bounding box polygon of the clipping plane")
        , grainColSlot("grainCol", "The colour used for the crystalites")
        , correctPBCSlot("correctPBC", "Activate correct handling of periodic boundary conditions") {

    this->clipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();

    this->lightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();

    this->grainCol[0] = this->grainCol[1] = this->grainCol[2] = 192.0f / 255.0f;
    this->grainColSlot << new core::param::StringParam(
        core::utility::ColourParser::ToString(this->grainCol[0], this->grainCol[1], this->grainCol[2]).PeekBuffer());

    this->showClipPlanePolySlot << new core::param::BoolParam(true);

    this->correctPBCSlot << new core::param::BoolParam(true);
}


/*
 * AbstractQuartzRenderer::~AbstractQuartzRenderer
 */
AbstractQuartzRenderer::~AbstractQuartzRenderer(void) {}


/*
 * AbstractQuartzRenderer::assertGrainColour
 */
void AbstractQuartzRenderer::assertGrainColour(void) {
    if (this->grainColSlot.IsDirty()) {
        this->grainColSlot.ResetDirty();
        try {
            core::utility::ColourParser::FromString(
                this->grainColSlot.Param<core::param::StringParam>()->Value().c_str(), this->grainCol[0],
                this->grainCol[1], this->grainCol[2]);
        } catch (...) {}
    }
}


/*
 * AbstractQuartzRenderer::getClipPlaneData
 */
core::view::CallClipPlane* AbstractQuartzRenderer::getClipPlaneData(void) {
    core::view::CallClipPlane* ccp = this->clipPlaneSlot.CallAs<core::view::CallClipPlane>();
    if (ccp != NULL) {
        if (!(*ccp)()) {
            ccp = NULL;
        }
    }
    return ccp;
}

} // namespace demos_gl
} /* end namespace megamol */
