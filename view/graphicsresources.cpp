/*
 * graphicsresources.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "view/graphicsresources.h"
#include "vislib/OutlineFont.h"
#include "vislib/Verdana.inc"

namespace megamol {
namespace core {
namespace view {

    /*
     * __openGLInfoFont
     */
    static const vislib::graphics::AbstractFont& __openGLVerdanaOutline(void) {
        const static vislib::graphics::gl::OutlineFont font(
            vislib::graphics::gl::FontInfo_Verdana,
            vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL_AND_OUTLINE);
        return font;
    }

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

/*
 * megamol::core::view::GetGlobalFont
 */
MEGAMOLCORE_API const vislib::graphics::AbstractFont&
megamol::core::view::GetGlobalFont(megamol::core::view::FontPurpose purpose) {
    switch (purpose) {
        case FONTPURPOSE_OPENGL_DEFAULT: return __openGLVerdanaOutline();
        case FONTPURPISE_OPENGL_INFO_HQ: return __openGLVerdanaOutline();
        default: return __openGLVerdanaOutline();
    }
}
