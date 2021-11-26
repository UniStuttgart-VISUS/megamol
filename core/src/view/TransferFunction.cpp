/*
 * TransferFunction.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/TransferFunction.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "stdafx.h"


using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::param;


view::TransferFunction::TransferFunction(void) : Module(), AbstractTransferFunction() {

    CallGetTransferFunctionDescription cgtfd;
    this->getTFSlot.SetCallback(cgtfd.ClassName(), cgtfd.FunctionName(0), &TransferFunction::requestTF);
    this->MakeSlotAvailable(&this->getTFSlot);

    this->tfParam << new TransferFunctionParam("");
    this->MakeSlotAvailable(&this->tfParam);
}


bool TransferFunction::requestTF(core::Call& call) {

    CallGetTransferFunction* cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr)
        return false;

    if (this->tfParam.IsDirty()) {
        this->tfParam.ResetDirty();

        // Check if range of initially loaded project value should be ignored
        auto tf_param_value = this->tfParam.Param<TransferFunctionParam>()->Value();
        bool tmp_ignore_project_range = TransferFunctionParam::IgnoreProjectRange(tf_param_value);

        // Update changed range propagated from the module via the call
        if (tmp_ignore_project_range && cgtf->ConsumeRangeUpdate()) {
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
        }

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

        ++this->version;
    }

    cgtf->SetTexture(this->texSize, this->tex.data(), this->texFormat, this->range, this->version);

    return true;
}
