#pragma once

#include <vector>

#include "datatools/AbstractParticleManipulator.h"

namespace megamol::datatools {

class ColorToDir : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ColorToDir";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Makes color values accessible as directions";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    ColorToDir();

    ~ColorToDir() override;

private:
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

    std::vector<std::vector<float>> data_;

    int frame_id_ = -1;

    std::size_t in_data_hash_ = std::numeric_limits<std::size_t>::max();

    std::size_t out_data_hash_ = 0;
};

} // namespace megamol::datatools
