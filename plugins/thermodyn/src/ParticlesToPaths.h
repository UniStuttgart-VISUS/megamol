#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include <unordered_map>

namespace megamol {
namespace thermodyn {

class ParticlesToPaths : public megamol::core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "ParticlesToPaths"; }

    /** Return module class description */
    static const char* Description(void) { return "Computes a particle pathlines from a set of particles"; }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    ParticlesToPaths();

    virtual ~ParticlesToPaths();

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    /** input of particle data */
    core::CallerSlot dataInSlot_;

    /** output of particle pathlines */
    core::CalleeSlot dataOutSlot_;

    int entrySize_;

    std::vector<std::unordered_map<uint64_t, std::vector<float>>> pathStore_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();
}; // end class ParticlesToPaths

} // end namespace thermodyn
} // end namespace megamol
