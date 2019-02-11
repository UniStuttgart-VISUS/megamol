#pragma once

#include <unordered_map>

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"


namespace megamol {
namespace thermodyn {

class PathFilter : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "PathFilter"; }

    /** Return module class description */
    static const char* Description(void) { return "Filter a particle pathlines"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    PathFilter();

    virtual ~PathFilter();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    /** input of particle pathlines */
    core::CallerSlot dataInSlot_;

    /** output of a subset of particle pathlines */
    core::CalleeSlot dataOutSlot_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    std::vector<int> entrySizes_;

    std::vector<bool> colsPresent_;

    std::vector<bool> dirsPresent_;

    std::vector<std::unordered_map<uint64_t, std::vector<float>>> pathStore_;
};

} // end namespace thermodyn
} // end namespace megamol
