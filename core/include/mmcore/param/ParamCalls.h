#pragma once

#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

namespace megamol::core::param {

template<typename ParamType>
class ParamCall : public GenericVersionedCall<typename ParamType::Value_T, EmptyMetaData> {
public:
    using Param = ParamType;
};

class FloatParamCall : public ParamCall<FloatParam> {
public:
    static const char* ClassName() {
        return "FloatParamCall";
    }

    static const char* Description() {
        return "FloatParamCall";
    }
};

using FloatParamCallDescription = core::factories::CallAutoDescription<FloatParamCall>;

class IntParamCall : public ParamCall<IntParam> {
public:
    static const char* ClassName() {
        return "IntParamCall";
    }

    static const char* Description() {
        return "IntParamCall";
    }
};

using IntParamCallDescription = core::factories::CallAutoDescription<IntParamCall>;
} // namespace megamol::core::param
