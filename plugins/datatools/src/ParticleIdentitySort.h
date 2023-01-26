#pragma once

#include "datatools/AbstractParticleManipulator.h"

namespace megamol {
namespace datatools {

class ParticleIdentitySort : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleIdentitySort";
    }

    /** Return module class description */
    static const char* Description() {
        return "Sorts particles according to the values stored in identity";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleIdentitySort();

    /** Dtor */
    ~ParticleIdentitySort() override;

protected:
    /**
     * Manipulates the particle data
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    std::vector<std::vector<char>> data_;
};

} /* end namespace datatools */
} /* end namespace megamol */
