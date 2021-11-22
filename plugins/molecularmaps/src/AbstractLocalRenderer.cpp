/*
 * AbstractLocalRenderer.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "AbstractLocalRenderer.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::molecularmaps;

/*
 * AbstractLocalRenderer::AbstractLocalRenderer
 */
AbstractLocalRenderer::AbstractLocalRenderer(void) {
    this->lastDataHash = 0;
}


/*
 * AbstractLocalRenderer::AbstractLocalRenderer
 */
AbstractLocalRenderer::~AbstractLocalRenderer(void) {
    // intentionally empty
}

/*
 * AbstractLocalRenderer::Release
 */
void AbstractLocalRenderer::Release(void) {
    this->release();
}
