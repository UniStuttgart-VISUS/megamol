#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/generic/CallGeneric.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

namespace megamol::core::param {

template<typename DataType, typename MetaDataType>
class AbstractParamCall : public GenericCall<DataType, MetaDataType> {
public:
    static unsigned int FunctionCount() {
        return 2;
    }

    static const unsigned int CallGetData = 0;

    static const unsigned int CallSetData = 1;

    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case CallGetData:
            return "GetData";
        case CallSetData:
            return "SetData";
        }
        return NULL;
    }
};

template<typename ParamType>
class ParamCall : public AbstractParamCall<typename ParamType::Value_T, EmptyMetaData> {
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
