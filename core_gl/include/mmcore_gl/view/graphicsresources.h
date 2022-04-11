/*
 * graphicsresources.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_GRAPHICSRESOURCES_H_INCLUDED
#define MEGAMOLCORE_GRAPHICSRESOURCES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/AbstractFont.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace view {


/**
 * Possible font purposes
 */
enum FontPurpose { FONTPURPOSE_OPENGL_DEFAULT, FONTPURPISE_OPENGL_INFO_HQ };


/**
 * Answers the OpenGL font to be used for the specific purpose
 *
 * @param purpose The purpose to font will be used for
 */
const vislib::graphics::AbstractFont& GetGlobalFont(FontPurpose purpose);


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_GRAPHICSRESOURCES_H_INCLUDED */
