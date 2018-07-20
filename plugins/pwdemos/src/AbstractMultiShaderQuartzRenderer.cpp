/*
 * AbstractMultiShaderQuartzRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractMultiShaderQuartzRenderer.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"

namespace megamol {
namespace demos {


/*
 * AbstractMultiShaderQuartzRenderer::AbstractMultiShaderQuartzRenderer
 */
AbstractMultiShaderQuartzRenderer::AbstractMultiShaderQuartzRenderer(void) : AbstractQuartzRenderer(),
        cntShaders(0), shaders(NULL), errShader() {
    // intentionally empty
}


/*
 * AbstractMultiShaderQuartzRenderer::~AbstractMultiShaderQuartzRenderer
 */
AbstractMultiShaderQuartzRenderer::~AbstractMultiShaderQuartzRenderer(void) {
    this->releaseShaders();
}


/*
 * AbstractMultiShaderQuartzRenderer::getCrystaliteData
 */
CrystalDataCall *AbstractMultiShaderQuartzRenderer::getCrystaliteData(void) {
    CrystalDataCall *tdc = AbstractQuartzRenderer::getCrystaliteData();
    if (tdc != NULL) {
        if ((tdc->DataHash() == 0) || (this->typesDataHash != tdc->DataHash())) {
            this->releaseShaders();
            ASSERT(this->shaders == NULL);
            this->cntShaders = tdc->GetCount();
            this->shaders = new vislib::graphics::gl::GLSLShader*[this->cntShaders];
            ::memset(this->shaders, 0, sizeof(vislib::graphics::gl::GLSLShader*) * this->cntShaders);
            this->typesDataHash = tdc->DataHash();
        }
    } else if (this->shaders != NULL) {
        this->releaseShaders();
    }
    return tdc;
}


/*
 * AbstractMultiShaderQuartzRenderer::ReleaseShaders
 */
void AbstractMultiShaderQuartzRenderer::releaseShaders(void) {
    if (this->shaders != NULL) {
        for (unsigned int i = 0; i < this->cntShaders; i++) {
            delete this->shaders[i];
        }
        this->cntShaders = 0;
        ARY_SAFE_DELETE(this->shaders);
    }
}

} /* end namespace demos */
} /* end namespace megamol */