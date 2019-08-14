/*
 * TransferFunction.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/TransferFunction.h"

#include "mmcore/param/TransferFunctionParam.h"


using namespace megamol::core;
using namespace megamol::core::view;


/*
 * TransferFunction::TransferFunction
 */
TransferFunction::TransferFunction(void)
    : Module()
    , getTFSlot("gettransferfunction", "Provides the transfer function")
    , tfParam("TransferFunction", "The transfer function serialized as JSON string.")
    , texID(0)
    , texSize(1)
    , tex()
    , texFormat(CallGetTransferFunction::TEXTURE_FORMAT_RGBA)
    , interpolMode(param::TransferFunctionParam::InterpolationMode::LINEAR)
    , range({0.0f, 1.0f}) {

    CallGetTransferFunctionDescription cgtfd;
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(0), &TransferFunction::requestTF);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->tfParam << new param::TransferFunctionParam("");
    this->MakeSlotAvailable(&this->tfParam);
}


/*
 * TransferFunction::~TransferFunction
 */
TransferFunction::~TransferFunction(void) { this->Release(); }


/*
 * TransferFunction::create
 */
bool TransferFunction::create(void) {

    return true;
}


/*
 * TransferFunction::release
 */
void TransferFunction::release(void) {

    glDeleteTextures(1, &this->texID);
    this->texID = 0;
}


/*
 * TransferFunction::requestTF
 */
bool TransferFunction::requestTF(Call& call) {

    CallGetTransferFunction* cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr) return false;

    if ((this->texID == 0) || this->tfParam.IsDirty()) {
        this->tfParam.ResetDirty();

        // Get current values from parameter string (Values are checked, too).
        param::TransferFunctionParam::TFNodeType tfnodes;
        if (!megamol::core::param::TransferFunctionParam::ParseTransferFunction(
            this->tfParam.Param<param::TransferFunctionParam>()->Value(), tfnodes, this->interpolMode, this->texSize, this->range)) {
            return false;
        }

        // Apply interpolation and generate texture data.
        if (this->interpolMode == param::TransferFunctionParam::InterpolationMode::LINEAR) {
            param::TransferFunctionParam::LinearInterpolation(this->tex, this->texSize, tfnodes);
        }
        else if (this->interpolMode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
            param::TransferFunctionParam::GaussInterpolation(this->tex, this->texSize, tfnodes);
        }

        if (this->texID != 0) {
            glDeleteTextures(1, &this->texID);
        }

        bool t1de = (glIsEnabled(GL_TEXTURE_1D) == GL_TRUE);
        if (!t1de) glEnable(GL_TEXTURE_1D);
        if (this->texID == 0) glGenTextures(1, &this->texID);

        GLint otid = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_1D, &otid);
        glBindTexture(GL_TEXTURE_1D, (GLuint)this->texID);

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, this->texSize, 0, GL_RGBA, GL_FLOAT, this->tex.data());

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);

        glBindTexture(GL_TEXTURE_1D, otid);

        if (!t1de) glDisable(GL_TEXTURE_1D);
        ++this->version;
    }
    cgtf->SetTexture(this->texID, this->texSize, this->tex.data(), this->texFormat,
        this->range, this->version);

    return true;
}
