#pragma once

#include <string>
#include <vector>

#include "mmcore/AbstractGetData3DCall.h"

#include "vislib/math/Cuboid.h"

#include "thermodyn.h"


namespace megamol {
namespace thermodyn {

class thermodyn_API BoxDataCall : public core::AbstractGetData3DCall {
public:
    struct box_entry {
        vislib::math::Cuboid<float> box_;

        std::string name_;

        float color_[4];
    };
    using box_entry_t = box_entry;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "BoxDataCall"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Transports boxes."; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractGetData3DCall::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractGetData3DCall::FunctionName(idx); }

    std::vector<box_entry_t>* GetBoxes() const { return boxes_; }

    void SetBoxes(std::vector<box_entry_t>* const boxes) { boxes_ = boxes; }

private:
    std::vector<box_entry_t>* boxes_ = nullptr;
}; // end class BoxDataCall

/** Call Descriptor.  */
typedef core::factories::CallAutoDescription<BoxDataCall> BoxDataCallDescription;

} // end namespace thermodyn
} // end namespace megamol
