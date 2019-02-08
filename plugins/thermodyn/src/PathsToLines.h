#pragma once

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "geometry_calls/LinesDataCall.h"

namespace megamol {
namespace thermodyn {

class PathToLines : public megamol::core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PathToLines"; }

    /** Return module class description */
    static const char* Description(void) { return "Transformes pathlines to lines"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathToLines();

    virtual ~PathToLines();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    core::CallerSlot dataInSlot_;

    core::CalleeSlot dataOutSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    using Lines = geocalls::LinesDataCall::Lines;

    std::vector<Lines> linesStore_;

    std::vector<std::vector<float>> linesData_;

}; // end class PathToLines

} // end namespace thermodyn
} // end namespace megamol
