/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/param/ParamCalls.h"

namespace megamol::core::param {
template<typename T>
class GenericParamModule : public core::Module {
public:
    GenericParamModule();

    ~GenericParamModule() override;

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

protected:
    bool create() override;

    void release() override;

private:
    bool get_data_cb(core::Call& c);

    bool set_data_cb(core::Call& c);

    core::CalleeSlot value_out_slot_;

    core::param::ParamSlot param_;

    uint64_t version_;

    typename T::Param::Value_T val_;
};

template<typename T>
inline GenericParamModule<T>::GenericParamModule() : value_out_slot_("valueOut", "")
                                                   , param_("param", "") {
    value_out_slot_.SetCallback(T::ClassName(), T::FunctionName(T::CallGetData), &GenericParamModule::get_data_cb);
    value_out_slot_.SetCallback(T::ClassName(), T::FunctionName(T::CallSetData), &GenericParamModule::set_data_cb);
    MakeSlotAvailable(&value_out_slot_);

    param_ << new typename T::Param(0);
    MakeSlotAvailable(&param_);
}

template<typename T>
inline GenericParamModule<T>::~GenericParamModule() {
    this->Release();
}

template<typename T>
inline bool GenericParamModule<T>::create() {
    return true;
}

template<typename T>
inline void GenericParamModule<T>::release() {}

template<typename T>
inline bool GenericParamModule<T>::get_data_cb(core::Call& c) {
    auto outCall = dynamic_cast<T*>(&c);
    if (outCall == nullptr)
        return false;

    if (param_.IsDirty()) {
        val_ = param_.Param<typename T::Param>()->Value();
        ++version_;
        param_.ResetDirty();
    }

    outCall->setData(val_, version_);

    return true;
}

template<typename T>
inline bool GenericParamModule<T>::set_data_cb(core::Call& c) {
    auto outCall = dynamic_cast<T*>(&c);
    if (outCall == nullptr)
        return false;

    val_ = outCall->getData();
    param_.Param<typename T::Param>()->SetValue(val_);

    return true;
}

class FloatParamModule : public GenericParamModule<FloatParamCall> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "FloatParamModule";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "FloatParamModule";
    }
};

class IntParamModule : public GenericParamModule<IntParamCall> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "IntParamModule";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "IntParamModule";
    }
};

} // namespace megamol::core::param
