/*
 * AbstractTexQuartzRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "AbstractTexQuartzRenderer.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <vector>

namespace megamol {
namespace demos {

/*
 * AbstractTexQuartzRenderer::AbstractTexQuartzRenderer
 */
AbstractTexQuartzRenderer::AbstractTexQuartzRenderer(void)
    : AbstractQuartzRenderer(), typeTexture(0) {
}


/*
 * AbstractTexQuartzRenderer::~AbstractTexQuartzRenderer
 */
AbstractTexQuartzRenderer::~AbstractTexQuartzRenderer(void) {
}


/*
 * AbstractTexQuartzRenderer::assertTypeTexture
 */
void AbstractTexQuartzRenderer::assertTypeTexture(CrystalDataCall& types) {
    if ((this->typesDataHash != 0) && (this->typesDataHash == types.DataHash())) return; // all up to date
    this->typesDataHash = types.DataHash();

    if (types.GetCount() == 0) {
        ::glDeleteTextures(1, &this->typeTexture);
        this->typeTexture = 0;
        return;
    }
    if (this->typeTexture == 0) {
        ::glGenTextures(1, &this->typeTexture);
    }

    unsigned mfc = 0;
    for (unsigned int i = 0; i < types.GetCount(); i++) {
        if (mfc < types.GetCrystals()[i].GetFaceCount()) {
            mfc = types.GetCrystals()[i].GetFaceCount();
        }
    }

    std::vector<float> dat;
    dat.resize(types.GetCount() * mfc * 4);

    for (unsigned int y = 0; y < types.GetCount(); y++) {
        const CrystalDataCall::Crystal& c = types.GetCrystals()[y];
        unsigned int x;
        for (x = 0; x < c.GetFaceCount(); x++) {
            vislib::math::Vector<float, 3> f = c.GetFace(x);
            dat[(x + y * mfc) * 4 + 3] = f.Normalise();
            dat[(x + y * mfc) * 4 + 0] = f.X();
            dat[(x + y * mfc) * 4 + 1] = f.Y();
            dat[(x + y * mfc) * 4 + 2] = f.Z();
        }
        for (; x < mfc; x++) {
            dat[(x + y * mfc) * 4 + 0] = 0.0f;
            dat[(x + y * mfc) * 4 + 1] = 0.0f;
            dat[(x + y * mfc) * 4 + 2] = 0.0f;
            dat[(x + y * mfc) * 4 + 3] = 0.0f;
        }
    }

    ::glBindTexture(GL_TEXTURE_2D, this->typeTexture);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    GLint initial_pack_alignment = 0;
    ::glGetIntegerv(GL_PACK_ALIGNMENT, &initial_pack_alignment);
    ::glPixelStorei(GL_PACK_ALIGNMENT, 1);
    
    ::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mfc, types.GetCount(), 0, GL_RGBA, GL_FLOAT, dat.data());
    ::glBindTexture(GL_TEXTURE_2D, 0);

    ::glPixelStorei(GL_PACK_ALIGNMENT, initial_pack_alignment);

}


/*
 * AbstractTexQuartzRenderer::releaseTypeTexture
 */
void AbstractTexQuartzRenderer::releaseTypeTexture(void) {
    ::glDeleteTextures(1, &this->typeTexture);
    this->typeTexture = 0;
}

} /* end namespace demos */
} /* end namespace megamol */