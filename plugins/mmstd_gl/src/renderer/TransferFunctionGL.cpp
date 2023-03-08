/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/renderer/TransferFunctionGL.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/TransferFunctionParam.h"


using namespace megamol::mmstd_gl;
using namespace megamol::core::param;


TransferFunctionGL::TransferFunctionGL(void)
        : ModuleGL()
        , AbstractTransferFunction()
        , texID(0)
        , interpolationParam("textureInterpolation", "Interpolation mode for the transfer function texture.") {

    CallGetTransferFunctionGLDescription cgtfd;
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(0), &TransferFunctionGL::requestTF);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->interpolationParam << new EnumParam(0);
    this->interpolationParam.Param<EnumParam>()->SetTypePair(0, "Linear");
    this->interpolationParam.Param<EnumParam>()->SetTypePair(1, "Nearest neighbor");
    this->MakeSlotAvailable(&this->interpolationParam);

    this->tfParam << new TransferFunctionParam("");
    this->MakeSlotAvailable(&this->tfParam);
}


bool TransferFunctionGL::create(void) {

    return true;
}


void TransferFunctionGL::release(void) {

    glDeleteTextures(1, &this->texID);
    this->texID = 0;
}


bool TransferFunctionGL::requestTF(core::Call& call) {

    auto cgtf = dynamic_cast<CallGetTransferFunctionGL*>(&call);
    if (cgtf == nullptr)
        return false;

    // update transfer function if still uninitialized
    bool something_has_changed = (this->texID == 0);

    // update transfer function if tf param is dirty
    if (this->tfParam.IsDirty()) {
        // Check if range of initially loaded project value should be ignored
        auto tf_param_value = this->tfParam.Param<TransferFunctionParam>()->Value();
        this->ignore_project_range = TransferFunctionParam::IgnoreProjectRange(tf_param_value);
        this->tfParam.ResetDirty();
        something_has_changed = true;
    }

    if (this->interpolationParam.IsDirty()) {
        this->interpolationParam.ResetDirty();
        something_has_changed = true;
    }

    // update transfer function if call asks for range update and range from project file is ignored
    if (cgtf->UpdateRange() && this->ignore_project_range) {
        // Update changed range propagated from the module via the call
        if (cgtf->ConsumeRangeUpdate()) {
            auto tf_param_value = this->tfParam.Param<TransferFunctionParam>()->Value();
            auto tmp_range = this->range;
            auto tmp_interpol = this->interpolMode;
            auto tmp_tex_size = this->texSize;
            TransferFunctionParam::NodeVector_t tmp_nodes;

            if (TransferFunctionParam::GetParsedTransferFunctionData(
                    tf_param_value, tmp_nodes, tmp_interpol, tmp_tex_size, tmp_range)) {
                // Set transfer function parameter value using updated range
                std::string tf_str;
                if (TransferFunctionParam::GetDumpedTransferFunction(
                        tf_str, tmp_nodes, tmp_interpol, tmp_tex_size, cgtf->Range())) {
                    this->tfParam.Param<TransferFunctionParam>()->SetValue(tf_str);
                }
            }
            something_has_changed = true;
        }
    }

    if (something_has_changed) {
        // Get current values from parameter string (Values are checked, too).
        TransferFunctionParam::NodeVector_t tmp_nodes;
        if (!TransferFunctionParam::GetParsedTransferFunctionData(this->tfParam.Param<TransferFunctionParam>()->Value(),
                tmp_nodes, this->interpolMode, this->texSize, this->range)) {
            return false;
        }

        // Apply interpolation and generate texture data.
        if (this->interpolMode == TransferFunctionParam::InterpolationMode::LINEAR) {
            this->tex = TransferFunctionParam::LinearInterpolation(this->texSize, tmp_nodes);
        } else if (this->interpolMode == TransferFunctionParam::InterpolationMode::GAUSS) {
            this->tex = TransferFunctionParam::GaussInterpolation(this->texSize, tmp_nodes);
        }

        if (this->texID != 0) {
            glDeleteTextures(1, &this->texID);
        }

        bool t1de = (glIsEnabled(GL_TEXTURE_1D) == GL_TRUE);
        if (!t1de)
            glEnable(GL_TEXTURE_1D);
        if (this->texID == 0)
            glGenTextures(1, &this->texID);

        GLint otid = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_1D, &otid);
        glBindTexture(GL_TEXTURE_1D, (GLuint)this->texID);
        const auto tex_format =
            this->texFormat == core::view::AbstractCallGetTransferFunction::TEXTURE_FORMAT_RGBA ? GL_RGBA : GL_RGB;

        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, this->texSize, 0, tex_format, GL_FLOAT, this->tex.data());

        // Always keep both at linear! UV-Coords here are data values, so OpenGL
        // cannot correctly decide if min or mag is the correct operation.
        if (this->interpolationParam.Param<EnumParam>()->Value() == 0) {
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        } else {
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_1D, otid);

        if (!t1de)
            glDisable(GL_TEXTURE_1D);
        ++this->version;
    }

    cgtf->SetTexture(this->texID, this->texSize, this->tex.data(), this->texFormat, this->range, this->version);

    return true;
}
