/*
 * TransferFunctionQuery.cpp
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "datatools_gl/TransferFunctionQuery.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "vislib/assert.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

using namespace megamol::datatools_gl;


/*
 * TransferFunctionQuery::TransferFunctionQuery
 */
TransferFunctionQuery::TransferFunctionQuery()
        : getTFSlot("gettransferfunction", "Connects to the transfer function module")
        , texDat()
        , texDatSize(0) {
    this->getTFSlot.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
}


/*
 * TransferFunctionQuery::~TransferFunctionQuery
 */
TransferFunctionQuery::~TransferFunctionQuery() {
    this->texDat.EnforceSize(0);
    this->texDatSize = 0;
}


/*
 * TransferFunctionQuery::Query
 */
void TransferFunctionQuery::Query(float* col, float val) {
    const size_t col_size = 4 * sizeof(float);

    if (this->texDatSize < 2) {
        // fetch transfer function
        mmstd_gl::CallGetTransferFunctionGL* cgtf = this->getTFSlot.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
        if ((cgtf != nullptr) && ((*cgtf)(0))) {
            ::glGetError();
            ::glEnable(GL_TEXTURE_1D);
            ::glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
            int texSize = 0;
            ::glGetTexLevelParameteriv(GL_TEXTURE_1D, 0, GL_TEXTURE_WIDTH, &texSize);
            if (::glGetError() == GL_NO_ERROR) {
                this->texDat.EnforceSize(texSize * col_size);
                ::glGetTexImage(GL_TEXTURE_1D, 0, GL_RGBA, GL_FLOAT, this->texDat.As<void>());
                if (::glGetError() != GL_NO_ERROR) {
                    this->texDat.EnforceSize(0);
                }
            }
            ::glBindTexture(GL_TEXTURE_1D, 0);
            ::glDisable(GL_TEXTURE_1D);
        }
        this->texDatSize = 2;
        if (texDat.GetSize() < 2 * col_size) {
            texDat.EnforceSize(2 * col_size);
            *texDat.AsAt<float>(0) = 0.0f;
            *texDat.AsAt<float>(4) = 0.0f;
            *texDat.AsAt<float>(8) = 0.0f;
            *texDat.AsAt<float>(12) = 1.0f;
            *texDat.AsAt<float>(16) = 1.0f;
            *texDat.AsAt<float>(20) = 1.0f;
            *texDat.AsAt<float>(24) = 1.0f;
            *texDat.AsAt<float>(28) = 1.0f;
        } else {
            texDatSize = static_cast<unsigned int>(texDat.GetSize() / col_size);
        }
        texDatSize--;
    }
    ASSERT(texDatSize >= 1);

    if (val < 0.0) {
        ::memcpy(col, texDat.At(0), col_size);
        return;
    }

    val *= static_cast<float>(texDatSize);
    unsigned int valIdx = static_cast<unsigned int>(val);
    if (valIdx >= texDatSize) {
        ::memcpy(col, texDat.At(texDatSize * col_size), col_size);
        return;
    }

    val -= static_cast<float>(valIdx);
    if (val < 0.0f)
        val = 0.0f;
    if (val > 1.0f)
        val = 1.0f;

    vislib::math::ShallowPoint<float, 4> c(col);
    vislib::math::ShallowPoint<float, 4> c1(texDat.AsAt<float>(valIdx * col_size));
    vislib::math::ShallowPoint<float, 4> c2(texDat.AsAt<float>((valIdx + 1) * col_size));
    c = c1.Interpolate(c2, val);
}
