#ifndef MMSTD_DATATOOLS__PARTICLEFRAMEINTERPOLATOR_H_INCLUDED
#define MMSTD_DATATOOLS__PARTICLEFRAMEINTERPOLATOR_H_INCLUDED

#include <map>
#include <unordered_map>
#include <vector>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class ParticleFrameInterpolator : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "ParticleFrameInterpolator"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Interpolates between subsequent MPDC frames."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    ParticleFrameInterpolator(void);

    virtual ~ParticleFrameInterpolator(void);

protected:
    bool create(void);

    void release(void);

private:
#define CROWBAR_DIMS 11

    typedef vislib::math::Point<float, CROWBAR_DIMS> particle_t;

    typedef vislib::math::Vector<float, CROWBAR_DIMS> trajectory_t;

    typedef std::vector<std::unordered_map<size_t, size_t>> particleIDMap_t;

    bool getDataCB(core::Call& c);

    bool getExtentCB(core::Call& c);

    bool serialiseParticleList(const core::moldyn::SimpleSphericalParticles& part, std::vector<particle_t>& out);

    bool serialiseForOutput(void);

    bool interpolateOnTrajectories(float inter, std::vector<std::vector<particle_t>>& nextFrame);

    bool findTrajectories(std::vector<std::vector<particle_t>>& nextFrame);

    bool findTrajectories(
        std::vector<std::vector<particle_t>>& nextFrame, particleIDMap_t& startMap, particleIDMap_t& nextMap);

    bool enumerateTrajectories(const particle_t& start, const particle_t& end, const trajectory_t& startTraject,
        std::map<float, trajectory_t>& out);

    trajectory_t calcTrajectory(const particle_t& start, const particle_t& end);

    bool checkForTransition(core::Call& inMPDC, core::Call& outMPDC);

    core::CallerSlot dataInSlot;

    core::CallerSlot supplementalSlot;

    core::CalleeSlot dataOutSlot;

    //#########################
    std::vector<std::vector<trajectory_t>> lastTrajectories;

    std::vector<std::vector<particle_t>> startPoints;

    std::vector<std::vector<particle_t>> startSupp;

    std::vector<std::vector<trajectory_t>> currentTrajectories;

    particleIDMap_t startMap;

    particleIDMap_t nextMap;

    particleIDMap_t lastMap;

    // std::vector<std::vector<particle_t>> endPoints;

    unsigned int startFrameID;

    unsigned int endFrameID;

    std::vector<std::vector<particle_t>> outPoints;

    std::vector<std::vector<float>> outData;

    vislib::math::Cuboid<float> bbox;

    float currentTimeStamp;

    float requestedTime;

    bool initialized;

    core::param::ParamSlot doSortParam;

    core::param::ParamSlot directionalParam;

    // std::vector<std::vector<particle_t>> nextFrame;

    vislib::math::Vector<float, 3> slerp(
        const float inter, const vislib::math::Vector<float, 3>& a, const vislib::math::Vector<float, 3>& b);

    void mergeInputs(
        std::vector<std::vector<particle_t>>& a, std::vector<std::vector<particle_t>>& b, particleIDMap_t& m);
}; // end class ParticleFrameInterpolator

} // namespace datatools
} // namespace stdplugin
} // end namespace megamol

#endif // end ifndef MMSTD_DATATOOLS__PARTICLEFRAMEINTERPOLATOR_H_INCLUDED
