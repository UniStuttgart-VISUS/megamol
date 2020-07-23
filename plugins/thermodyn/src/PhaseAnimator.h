#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace thermodyn {

class PhaseAnimator : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PhaseAnimator"; }

    /** Return module class description */
    static const char* Description(void) { return "Helper module animating phases"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PhaseAnimator();

    virtual ~PhaseAnimator();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    bool isDirty() {
        return fluid_alpha_slot_.IsDirty() || interface_alpha_slot_.IsDirty() || gas_alpha_slot_.IsDirty();
    }

    void resetDirty() {
        fluid_alpha_slot_.ResetDirty();
        interface_alpha_slot_.ResetDirty();
        gas_alpha_slot_.ResetDirty();
    }

    core::CalleeSlot out_data_slot_;

    core::CallerSlot part_in_data_slot_;

    core::CallerSlot box_in_data_slot_;

    core::param::ParamSlot fluid_alpha_slot_;

    core::param::ParamSlot interface_alpha_slot_;

    core::param::ParamSlot gas_alpha_slot_;

    size_t data_hash_;

    size_t out_data_hash_;

    unsigned int frame_id_;

    std::vector<std::vector<float>> data_;

}; // end class PhaseAnimator

} // end namespace thermodyn
} // end namespace megamol