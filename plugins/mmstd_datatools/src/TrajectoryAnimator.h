#ifndef MEGAMOL_DATATOOLS_TRAJECTORYANIMATOR_H_INCLUDED
#define MEGAMOL_DATATOOLS_TRAJECTORYANIMATOR_H_INCLUDED

#include <vector>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/LinesDataCall.h"

#include "vislib/math/Cuboid.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class TrajectoryAnimator : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "TrajectoryAnimator"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Creates an animation to highlight trajectories crossing gas/fluid boundary";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** ctor */
    TrajectoryAnimator(void);

    /** dtor */
    virtual ~TrajectoryAnimator(void);

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);

private:
    struct point {
        float x;
        float y;
        float z;
    };

    static point& interpolate(point const& a, point const& b, float i) {
        point ret;
        ret.x = a.x*(1 - i) + b.x*i;
        ret.y = a.y*(1 - i) + b.y*i;
        ret.z = a.z*(1 - i) + b.z*i;
        return ret;
    }

    bool getLinesDataCallback(megamol::core::Call& c);

    bool getLinesExtentCallback(megamol::core::Call& c);

    bool getPointsDataCallback(megamol::core::Call& c);

    bool getPointsExtentCallback(megamol::core::Call& c);

    bool assertData(megamol::core::Call& linesC);

    bool isDirty();

    megamol::core::CalleeSlot linesOutSlot;

    megamol::core::CalleeSlot pointsOutSlot;

    megamol::core::CallerSlot linesInSlot;

    megamol::core::CallerSlot pointsInSlot;

    megamol::core::param::ParamSlot animationFactorSlot;

    megamol::core::param::ParamSlot inFrameCountSlot;

    megamol::core::param::ParamSlot globalRadiusSlot;

    vislib::math::Cuboid<float> bbox;

    unsigned int inFrameCount;

    unsigned int outFrameCount;

    size_t datahash;

    unsigned int frameID;

    unsigned int startFrameID;

    unsigned int endFrameID;

    std::vector<float> pointData;
}; /* end class TrajectoryAnimator */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_TRAJECTORYANIMATOR_H_INCLUDED */