/*
 * AbstractQuartzRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractQuartzRenderer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"

namespace megamol {
namespace demos {

/*
 * AbstractQuartzRenderer::AbstractQuartzRenderer
 */
AbstractQuartzRenderer::AbstractQuartzRenderer(void) : AbstractQuartzModule(),
        clipPlaneSlot("clipplane", "Slot connecting to the clipping plane provider"),
        showClipPlanePolySlot("showClipPlanePoly", "Shows/Hides the bounding box polygon of the clipping plane"),
        grainColSlot("grainCol", "The colour used for the crystalites"),
        correctPBCSlot("correctPBC", "Activate correct handling of periodic boundary conditions") {

    this->clipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();

    this->grainCol[0] = this->grainCol[1] = this->grainCol[2] = 192.0f / 255.0f;
    this->grainColSlot << new core::param::StringParam(
        core::utility::ColourParser::ToString(
        this->grainCol[0], this->grainCol[1], this->grainCol[2]));

    this->showClipPlanePolySlot << new core::param::BoolParam(true);

    this->correctPBCSlot << new core::param::BoolParam(true);

}


/*
 * AbstractQuartzRenderer::~AbstractQuartzRenderer
 */
AbstractQuartzRenderer::~AbstractQuartzRenderer(void) {
}


/*
 * AbstractQuartzRenderer::assertGrainColour
 */
void AbstractQuartzRenderer::assertGrainColour(void) {
    if (this->grainColSlot.IsDirty()) {
        this->grainColSlot.ResetDirty();
        try {
            core::utility::ColourParser::FromString(
                this->grainColSlot.Param<core::param::StringParam>()->Value(),
                this->grainCol[0], this->grainCol[1], this->grainCol[2]);
        } catch(...) {
        }
    }
}


/*
 * AbstractQuartzRenderer::getClipPlaneData
 */
core::view::CallClipPlane *AbstractQuartzRenderer::getClipPlaneData(void) {
    core::view::CallClipPlane *ccp = this->clipPlaneSlot.CallAs<core::view::CallClipPlane>();
    if (ccp != NULL) {
        if (!(*ccp)()) {
            ccp = NULL;
        }
    }
    return ccp;
}

} /* end namespace demos */
} /* end namespace megamol */