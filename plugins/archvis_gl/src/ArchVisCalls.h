/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "FEMModel.h"
#include "ScaleModel.h"
#include "mmstd/generic/CallGeneric.h"

namespace megamol::archvis_gl {

class CallFEMModel : public core::GenericVersionedCall<std::shared_ptr<FEMModel>, core::Spatial3DMetaData> {
public:
    inline CallFEMModel() : GenericVersionedCall<std::shared_ptr<FEMModel>, core::Spatial3DMetaData>() {}
    ~CallFEMModel() = default;

    static const char* ClassName(void) {
        return "CallFEMModel";
    }
    static const char* Description(void) {
        return "Call that transports...";
    }
};

class CallScaleModel : public core::GenericVersionedCall<std::shared_ptr<ScaleModel>, core::Spatial3DMetaData> {
public:
    inline CallScaleModel() : GenericVersionedCall<std::shared_ptr<ScaleModel>, core::Spatial3DMetaData>() {}
    ~CallScaleModel() = default;

    static const char* ClassName(void) {
        return "CallScaleModel";
    }
    static const char* Description(void) {
        return "Call that transports...";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<CallScaleModel> ScaleModelCallDescription;
typedef megamol::core::factories::CallAutoDescription<CallFEMModel> FEMModelCallDescription;

} // namespace megamol::archvis_gl
