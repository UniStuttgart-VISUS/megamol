/*
 * LinearTransferFunction.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/LinearTransferFunction.h"

#include "mmcore/param/LinearTransferFunctionParam.h"


using namespace megamol::core;
using namespace megamol::core::view;


/*
 * LinearTransferFunction::LinearTransferFunction
 */
LinearTransferFunction::LinearTransferFunction(void)
    : Module()
    , getTFSlot("gettransferfunction", "Provides the transfer function")
    , tfParam("TransferFunction", "The transfer function serialized as JSON string.")
    , texID(0)
    , texSize(1)
    , tex()
    , texFormat(CallGetTransferFunction::TEXTURE_FORMAT_RGB)
    , interpolMode(param::LinearTransferFunctionParam::InterpolationMode::LINEAR)
{

    CallGetTransferFunctionDescription cgtfd;
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(0), &LinearTransferFunction::requestTF);
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(1), &LinearTransferFunction::interfaceIsDirty);
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(2), &LinearTransferFunction::interfaceResetDirty);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->tfParam << new param::LinearTransferFunctionParam("");
    this->MakeSlotAvailable(&this->tfParam);
}


/*
 * LinearTransferFunction::~LinearTransferFunction
 */
LinearTransferFunction::~LinearTransferFunction(void) { this->Release(); }


/*
 * LinearTransferFunction::create
 */
bool LinearTransferFunction::create(void) {

    return true;
}


/*
 * LinearTransferFunction::release
 */
void LinearTransferFunction::release(void) {

    glDeleteTextures(1, &this->texID);
    this->texID = 0;    
}


/*
 * LinearTransferFunction::requestTF
 */
bool LinearTransferFunction::requestTF(Call& call) {

    CallGetTransferFunction* cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr) return false;

    if ((this->texID == 0) || this->tfParam.IsDirty()) {
        this->tfParam.ResetDirty();

        param::LinearTransferFunctionParam::TFDataType tfdata;

        // Get current values from parameter string. Values are checked, too.
        if (!megamol::core::param::LinearTransferFunctionParam::ParseTransferFunction(
            this->tfParam.Param<param::LinearTransferFunctionParam>()->Value(), tfdata, this->interpolMode, this->texSize)) {
            return false;
        }

        // Apply interpolation and generate texture data.
        if (this->interpolMode == param::LinearTransferFunctionParam::InterpolationMode::LINEAR) {
            param::LinearTransferFunctionParam::LinearInterpolation(this->tex, this->texSize, tfdata);
        }
        else if (this->interpolMode == param::LinearTransferFunctionParam::InterpolationMode::GAUSS) {
            param::LinearTransferFunctionParam::GaussInterpolation(this->tex, this->texSize, tfdata);
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
    }

    cgtf->SetTexture(this->texID, this->texSize, this->tex.data(), CallGetTransferFunction::TEXTURE_FORMAT_RGBA);

    return true;
}


/*
 * LinearTransferFunction::InterfaceIsDirty
 */
bool LinearTransferFunction::interfaceIsDirty(Call& call) {

    CallGetTransferFunction* cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr) return false;

    bool retval = tfParam.IsDirty();

    cgtf->setDirty(retval);
    return true;
}


/*
 * LinearTransferFunction::interfaceResetDirty
 */
bool LinearTransferFunction::interfaceResetDirty(Call& call) {

    CallGetTransferFunction* cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr) return false;

    tfParam.ResetDirty();
    return true;
}