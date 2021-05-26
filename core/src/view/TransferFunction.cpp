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
using namespace megamol::core::param;


TransferFunction::TransferFunction(void)
    : Module()
    , getTFSlot("gettransferfunction", "Provides the transfer function")
    , tfParam("TransferFunction", "The transfer function serialized as JSON string.")
    , texID(0)
    , texSize(1)
    , tex()
    , texFormat(CallGetTransferFunction::TEXTURE_FORMAT_RGBA)
    , interpolMode(TransferFunctionParam::InterpolationMode::LINEAR)
    , range({0.0f, 1.0f})
    , ignore_project_range(false)
    , version(0)
    , last_frame_id(0) {

    CallGetTransferFunctionDescription cgtfd;
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(0), &TransferFunction::requestTF);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->tfParam << new TransferFunctionParam("");
    this->MakeSlotAvailable(&this->tfParam);
}


TransferFunction::~TransferFunction(void) { this->Release(); }


bool TransferFunction::create(void) {

    return true;
}


void TransferFunction::release(void) {

    glDeleteTextures(1, &this->texID);
    this->texID = 0;
}


bool TransferFunction::requestTF(Call& call) {

    CallGetTransferFunction* cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr) return false;

    auto tmp_range = this->range;
    auto tmp_interpol = this->interpolMode;
    auto tmp_tex_size = this->texSize;
    TransferFunctionParam::NodeVector_t tmp_nodes;
    
    // Check if range of initially loaded project value should be ignored
    auto tf_param_value = this->tfParam.Param<TransferFunctionParam>()->Value();
    this->ignore_project_range = TransferFunctionParam::IgnoreProjectRange(tf_param_value);

    // Update changed range propagated from the module via the call
    if (this->ignore_project_range && cgtf->ConsumeRangeUpdate()) {
        if (TransferFunctionParam::GetParsedTransferFunctionData( tf_param_value, tmp_nodes, tmp_interpol, tmp_tex_size,tmp_range)) {
            // Set transfer function parameter value using updated range
            std::string tf_str;
            if (TransferFunctionParam::GetDumpedTransferFunction(tf_str, tmp_nodes, tmp_interpol, tmp_tex_size, cgtf->Range())) {
                this->tfParam.Param<TransferFunctionParam>()->SetValue(tf_str);
            }
        }
    }

    if ((this->texID == 0) || this->tfParam.IsDirty()) {
        this->tfParam.ResetDirty();

        // Get current values from parameter string (Values are checked, too).
        if (!TransferFunctionParam::GetParsedTransferFunctionData(this->tfParam.Param<TransferFunctionParam>()->Value(), tmp_nodes, this->interpolMode, this->texSize, this->range)) {
            return false;
        }

        // Apply interpolation and generate texture data.
        if (this->interpolMode == TransferFunctionParam::InterpolationMode::LINEAR) {
            this->tex = TransferFunctionParam::LinearInterpolation(this->texSize, tmp_nodes);
        }
        else if (this->interpolMode == TransferFunctionParam::InterpolationMode::GAUSS) {
            this->tex = TransferFunctionParam::GaussInterpolation(this->texSize, tmp_nodes);
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

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, this->texSize, 0, this->texFormat, GL_FLOAT, this->tex.data());

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
