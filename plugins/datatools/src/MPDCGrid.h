#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/MultiParticleDataCall.h"

#include "vislib/math/Point.h"


namespace megamol::datatools {

class MPDCGrid : public core::Module {
public:
    struct Particle {
        vislib::math::Point<float, 3> pos;
        unsigned char cr, cg, cb, ca;
    };

    struct Box {
        vislib::math::Point<float, 3> lower, upper;

        vislib::math::Vector<float, 3> span() const {
            return {upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]};
        }

        vislib::math::Point<float, 3> calc_center() const {
            return lower + span() / 2.0f;
        }
    };

    struct BrickLet {
        size_t begin, end;
        Box bounds;
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "MPDCGrid";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Converts a MPDC in to a regular grid (cells exposed as individual particle list entries)";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    MPDCGrid();

    ~MPDCGrid() override;

protected:
    bool create() override;

    void release() override;

private:
    bool getDataCallback(core::Call& c);

    bool getExtentCallback(core::Call& c);

    std::vector<BrickLet> gridify(
        std::vector<Particle>& particles, Box const& bbox, size_t maxSize, size_t begin, size_t end);

    std::vector<geocalls::SimpleSphericalParticles> separate(
        std::vector<Particle> const& particles, std::vector<BrickLet> const& bricks, float radius);

    core::CalleeSlot data_out_slot_;

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot max_size_slot_;

    size_t data_out_hash_;

    size_t data_in_hash_;

    int out_frame_id_;

    std::vector<std::vector<Particle>> data_;

    std::vector<geocalls::SimpleSphericalParticles> output_;
}; // class MPDCGrid

inline vislib::math::Vector<float, 3> span(vislib::math::Cuboid<float> const& box) {
    return vislib::math::Vector<float, 3>(box.Width(), box.Height(), box.Depth());
}

inline int arg_max(vislib::math::Vector<float, 3> const& vec) {
    if (vec.X() > vec.Y()) {
        if (vec.Z() > vec.X()) {
            return 2;
        } else {
            return 0;
        }
    } else {
        if (vec.Z() > vec.Y()) {
            return 2;
        } else {
            return 1;
        }
    }
}

} // namespace megamol::datatools
