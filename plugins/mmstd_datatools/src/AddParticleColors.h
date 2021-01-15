#pragma once

#include "mmcore/CallerSlot.h"

#include "mmstd_datatools/AbstractParticleManipulator.h"

#include "glm/glm.hpp"

namespace megamol::stdplugin::datatools {
class AddParticleColors : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "AddParticleColors";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Transforms COL_I to RGBA";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    AddParticleColors(void);

    /** Dtor */
    virtual ~AddParticleColors(void);

protected:
    bool manipulateData(core::moldyn::MultiParticleDataCall& outData, core::moldyn::MultiParticleDataCall& inData) override;

private:
    struct color {
        glm::vec4 rgba;
    };

    float lerp(float a, float b, float inter);

    glm::vec4 sample_tf(float const* tf, unsigned int tf_size, int base, float rest);

    core::CallerSlot _tf_slot;

    unsigned int _frame_id = std::numeric_limits<unsigned int>::max();

    std::size_t _in_data_hash = std::numeric_limits<std::size_t>::max();

    std::size_t _out_data_hash = 0;

    std::vector<std::vector<color>> _colors;
};
} // namespace megamol::stdplugin::datatools
