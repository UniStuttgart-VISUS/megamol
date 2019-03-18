/*
 * LinearTransferFunction.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/LinearTransferFunction.h"


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
 * LinearTransferFunction::LinearInterpolation
 */
void LinearTransferFunction::LinearInterpolation(std::vector<float> &out_texdata, unsigned int in_texsize, const param::LinearTransferFunctionParam::TFType &in_tfdata) {

    out_texdata.resize(4 * in_texsize);
    std::array<float, 5> cx1 = in_tfdata[0];
    std::array<float, 5> cx2 = in_tfdata[0];
    int p1 = 0;
    int p2 = 0;
    size_t data_cnt = in_tfdata.size();
    for (size_t i = 1; i < data_cnt; i++) {
        cx1 = cx2;
        p1 = p2;
        cx2 = in_tfdata[i];
        assert(cx2[4] <= 1.0f + 1e-5f); // 1e-5f = vislib::math::FLOAT_EPSILON
        p2 = static_cast<int>(cx2[4] * static_cast<float>(in_texsize - 1));
        assert(p2 < static_cast<int>(in_texsize));
        assert(p2 >= p1);

        for (int p = p1; p <= p2; p++) {
            float al = static_cast<float>(p - p1) / static_cast<float>(p2 - p1);
            float be = 1.0f - al;

            out_texdata[p * 4] = cx1[0] * be + cx2[0] * al;
            out_texdata[p * 4 + 1] = cx1[1] * be + cx2[1] * al;
            out_texdata[p * 4 + 2] = cx1[2] * be + cx2[2] * al;
            out_texdata[p * 4 + 3] = cx1[3] * be + cx2[3] * al;
        }
    }
}


/*
 * LinearTransferFunction::requestTF
 */
bool LinearTransferFunction::requestTF(Call& call) {

    CallGetTransferFunction* cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr) return false;

    bool dirty = this->tfParam.IsDirty();
    if ((this->texID == 0) || dirty) {
        this->tfParam.ResetDirty();

        param::LinearTransferFunctionParam::TFType tfdata;

        // Get current values from parameter string. Values are checked, too.
        if (!megamol::core::param::LinearTransferFunctionParam::ParseTransferFunction(
            this->tfParam.Param<param::LinearTransferFunctionParam>()->Value(), tfdata, this->interpolMode, this->texSize)) {
            return false;
        }

        // Apply interpolation and generate texture data.
        if (this->interpolMode == param::LinearTransferFunctionParam::InterpolationMode::LINEAR) {
            this->LinearInterpolation(this->tex, this->texSize, tfdata);
        }
        else if (this->interpolMode == param::LinearTransferFunctionParam::InterpolationMode::GAUSS) {
            // TODO: Implement ...
            return false;
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