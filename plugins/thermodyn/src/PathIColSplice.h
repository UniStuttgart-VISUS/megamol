#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "thermodyn/PathLineDataCall.h"

namespace megamol {
namespace thermodyn {

class PathIColSplice : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PathIColSplice"; }

    /** Return module class description */
    static const char* Description(void) { return "Computes a particle pathlines with values from ICol"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathIColSplice();

    virtual ~PathIColSplice();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    static std::vector<float> enlargeVector(std::vector<float> const& input, size_t const stride) {
        std::vector<float> ret(input.size() / stride * (stride + 1));
        for (size_t idx = 0; idx < input.size() / stride; ++idx) {
            for (size_t s = 0; s < stride; ++s) {
                ret[idx * (stride + 1) + s] = input[idx * stride + s];
            }
        }
        return ret;
    }

    core::CallerSlot pathsInSlot_;

    core::CallerSlot icolInSlot_;

    core::CalleeSlot dataOutSlot_;

    size_t inPathsHash_ = std::numeric_limits<size_t>::max();

    size_t inIColHash_ = std::numeric_limits<size_t>::max();

    size_t outDataHash_ = 0;

    PathLineDataCall::pathline_store_set_t outPathStore_;

    PathLineDataCall::entrysizes_t outEntrySizes_;

    PathLineDataCall::color_flags_t outColsPresent_;

    PathLineDataCall::dir_flags_t outDirsPresent_;
}; // end class PathIColSplice

} // end namespace thermodyn
} // end namespace megamol
