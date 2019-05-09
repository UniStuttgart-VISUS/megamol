#pragma once

#include <unordered_map>
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"

#include "thermodyn/PathLineDataCall.h"

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

    core::param::ParamSlot cyclXSlot;
    core::param::ParamSlot cyclYSlot;
    core::param::ParamSlot cyclZSlot;

    core::param::ParamSlot bboxSlot;

    core::param::ParamSlot projectionSlot_;

    core::param::ParamSlot frameSkipSlot_;

    std::vector<int> entrySizes_;

    std::vector<bool> colsPresent_;

    std::vector<bool> dirsPresent_;

    std::vector<std::unordered_map<uint64_t, std::vector<float>>> pathStore_;

    PathLineDataCall::pathline_frame_store_set_t pathFrameStore_;

    size_t inDataHash_ = std::numeric_limits<size_t>::max();

    vislib::math::Cuboid<float> bbox;
}; // end class ParticlesToPaths

} // end namespace thermodyn
} // end namespace megamol
