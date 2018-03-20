#ifndef MEGAMOL_DATATOOLS_TRAJECTORYANIMATOR_H_INCLUDED
#define MEGAMOL_DATATOOLS_TRAJECTORYANIMATOR_H_INCLUDED

#include <vector>
#include <unordered_map>

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

        bool operator==(point const& lhs) const {
            return x == lhs.x && y == lhs.y && z == lhs.z;
        }

        bool operator!=(point const& lhs) const {
            return x != lhs.x || y != lhs.y || z != lhs.z;
        }

        float operator[](size_t const idx) const {
            if (idx == 0) return x;
            if (idx == 1) return y;
            if (idx == 2) return z;
            return 0.0f;
        }
    };

    struct color {
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;

        bool operator==(color const& lhs) const {
            return r == lhs.r && g == lhs.g && b == lhs.b && a == lhs.a;
        }
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

    bool getHighlightsDataCallback(megamol::core::Call& c);

    bool getHighlightsExtentCallback(megamol::core::Call& c);

    bool assertData(megamol::core::Call& linesC);

    bool isDirty();

    static bool checkTransition(point const &pos_a, point const& pos_b, float const min_trans, float const max_trans, unsigned int const trans_dir,
                         float const trans_eps);

    std::pair<std::vector<float>, std::vector<unsigned char>> removePadding(megamol::geocalls::LinesDataCall::Lines const& l) const;

    megamol::core::CalleeSlot linesOutSlot;

    megamol::core::CalleeSlot pointsOutSlot;

    megamol::core::CalleeSlot highlightsOutSlot;

    megamol::core::CallerSlot linesInSlot;

    megamol::core::CallerSlot pointsInSlot;

    megamol::core::param::ParamSlot animationFactorSlot;

    megamol::core::param::ParamSlot inFrameCountSlot;

    megamol::core::param::ParamSlot globalRadiusSlot;

    megamol::core::param::ParamSlot animationLengthSlot;

    megamol::core::param::ParamSlot minTransSlot;

    megamol::core::param::ParamSlot maxTransSlot;

    megamol::core::param::ParamSlot transDirSlot;

    megamol::core::param::ParamSlot transEpsSlot;

    megamol::core::param::ParamSlot highlightRadiusSlot;

    vislib::math::Cuboid<float> bbox;

    unsigned int inFrameCount;

    unsigned int outFrameCount;

    size_t datahash;

    unsigned int frameID;

    unsigned int startFrameID;

    unsigned int endFrameID;

    std::vector<float> pointData;

    std::unordered_map<unsigned int /* line idx */, unsigned int /* start frame */> transitionLines2Frames;

    std::unordered_map<unsigned int, std::pair<std::vector<float>, std::vector<unsigned char>>> transitionLines;

    std::vector<megamol::geocalls::LinesDataCall::Lines> renderQueue;

    std::vector<float> highlightPointData;

    std::vector<float> dummyLinePos;

    megamol::geocalls::LinesDataCall::Lines dummyLine;
}; /* end class TrajectoryAnimator */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_TRAJECTORYANIMATOR_H_INCLUDED */