/*
 * TransferFunction.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmstd/renderer/TransferFunction.h"
#include "mmcore/param/TransferFunctionParam.h"


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

    auto cgtf = dynamic_cast<CallGetTransferFunction*>(&call);
    if (cgtf == nullptr)
        return false;

    // update transfer function if still uninitialized
    bool something_has_changed = false;

    // update transfer function if tf param is dirty
    if (tfParam.IsDirty()) {
        // Check if range of initially loaded project value should be ignored
        auto tf_param_value = this->tfParam.Param<TransferFunctionParam>()->Value();
        ignore_project_range = TransferFunctionParam::IgnoreProjectRange(tf_param_value);
        tfParam.ResetDirty();
        something_has_changed = true;
    }

    // update transfer function if call ask for range update range from project file is ignored
    if (cgtf->UpdateRange() && this->ignore_project_range) {
        // Update changed range propagated from the module via the call
        if (cgtf->ConsumeRangeUpdate()) {
            auto tf_param_value = tfParam.Param<TransferFunctionParam>()->Value();
            auto tmp_range = range;
            auto tmp_interpol = interpolMode;
            auto tmp_tex_size = texSize;
            TransferFunctionParam::NodeVector_t tmp_nodes;

            if (TransferFunctionParam::GetParsedTransferFunctionData(
                    tf_param_value, tmp_nodes, tmp_interpol, tmp_tex_size, tmp_range)) {
                // Set transfer function parameter value using updated range
                std::string tf_str;
                if (TransferFunctionParam::GetDumpedTransferFunction(
                        tf_str, tmp_nodes, tmp_interpol, tmp_tex_size, cgtf->Range())) {
                    tfParam.Param<TransferFunctionParam>()->SetValue(tf_str);
                    tfParam.ResetDirty();
                }
            }
            something_has_changed = true;
        }
    }

    if (something_has_changed) {
        // Get current values from parameter string (Values are checked, too).
        TransferFunctionParam::NodeVector_t tmp_nodes;
        if (!TransferFunctionParam::GetParsedTransferFunctionData(tfParam.Param<TransferFunctionParam>()->Value(),
                tmp_nodes, interpolMode, texSize, range)) {
            return false;
        }

        // Apply interpolation and generate texture data.
        if (interpolMode == TransferFunctionParam::InterpolationMode::LINEAR) {
            tex = TransferFunctionParam::LinearInterpolation(texSize, tmp_nodes);
        } else if (interpolMode == TransferFunctionParam::InterpolationMode::GAUSS) {
            tex = TransferFunctionParam::GaussInterpolation(texSize, tmp_nodes);
        }

        ++version;
    }

    cgtf->SetTexture(texSize, tex.data(), texFormat, range, version);

    return true;
}
