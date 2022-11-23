/*
 * AbstractMultiShaderQuartzRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "AbstractMultiShaderQuartzRenderer.h"

#include "OpenGL_Context.h"
#include "mmcore/utility/ColourParser.h"

namespace megamol::demos_gl {


/*
 * AbstractMultiShaderQuartzRenderer::AbstractMultiShaderQuartzRenderer
 */
AbstractMultiShaderQuartzRenderer::AbstractMultiShaderQuartzRenderer()
        : AbstractQuartzRenderer()
        , shaders()
        , errShader() {
    // intentionally empty
}


/*
 * AbstractMultiShaderQuartzRenderer::~AbstractMultiShaderQuartzRenderer
 */
AbstractMultiShaderQuartzRenderer::~AbstractMultiShaderQuartzRenderer() {
    this->releaseShaders();
}


/*
 * AbstractMultiShaderQuartzRenderer::getCrystaliteData
 */
CrystalDataCall* AbstractMultiShaderQuartzRenderer::getCrystaliteData() {
    CrystalDataCall* tdc = AbstractQuartzRenderer::getCrystaliteData();
    if (tdc != NULL) {
        if ((tdc->DataHash() == 0) || (this->typesDataHash != tdc->DataHash())) {
            this->releaseShaders();
            ASSERT(this->shaders.empty());
            this->shaders.resize(tdc->GetCount());
            this->typesDataHash = tdc->DataHash();
        }
    } else if (!this->shaders.empty()) {
        this->releaseShaders();
    }
    return tdc;
}


/*
 * AbstractMultiShaderQuartzRenderer::ReleaseShaders
 */
void AbstractMultiShaderQuartzRenderer::releaseShaders() {
    for (auto& s : shaders) {
        s.reset();
    }
    shaders.clear();
}

} // namespace megamol::demos_gl
