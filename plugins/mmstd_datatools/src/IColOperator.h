#pragma once

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/AbstractParticleManipulator.h"

namespace megamol::stdplugin::datatools {

class IColOperator : public AbstractParticleManipulator {
public:
    static const char* ClassName(void) {
        return "IColOperator";
    }
    static const char* Description(void) {
        return "Adds two ICol value streams:  c[] = a_s * a[] + b_s * b[]";
    }
    static bool IsAvailable(void) {
        return true;
    }

    IColOperator();
    virtual ~IColOperator();

protected:
    virtual bool manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData, megamol::core::moldyn::MultiParticleDataCall& inDataA);

private:
    enum class operator_type { ot_plus, ot_and, ot_less, ot_greater };

    using icol_operator = std::function<float(float, float)>;

    bool is_dirty() {
        return primary_operator_slot_.IsDirty() || a_operator_slot_.IsDirty() || b_operator_slot_.IsDirty() ||
               a_value_slot_.IsDirty() || b_value_slot_.IsDirty();
    }

    void reset_dirty() {
        primary_operator_slot_.ResetDirty();
        a_operator_slot_.ResetDirty();
        b_operator_slot_.ResetDirty();
        a_value_slot_.ResetDirty();
        b_value_slot_.ResetDirty();
    }

    icol_operator parse_icol_operator_type(operator_type type);

    core::CallerSlot in_dataB_slot_;
    //        core::param::ParamSlot aOffsetSlot;
    // core::param::ParamSlot aScaleSlot;
    //        core::param::ParamSlot bOffsetSlot;
    // core::param::ParamSlot bScaleSlot;

    core::param::ParamSlot primary_operator_slot_;
    core::param::ParamSlot a_operator_slot_;
    core::param::ParamSlot b_operator_slot_;
    core::param::ParamSlot a_value_slot_;
    core::param::ParamSlot b_value_slot_;

    size_t inAHash_;
    size_t inBHash_;
    size_t outHash_;
    unsigned int frameID_;
    std::vector<float> colors_;
    float minCol_, maxCol_;
};

} // namespace megamol::stdplugin::datatools
