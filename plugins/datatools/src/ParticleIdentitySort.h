#pragma once

#include "datatools/AbstractParticleManipulator.h"

namespace megamol {
namespace datatools {

class ParticleIdentitySort : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "ParticleIdentitySort";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Sorts particles according to the values stored in identity";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    ParticleIdentitySort(void);

    /** Dtor */
    virtual ~ParticleIdentitySort(void);

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
    virtual bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData);

private:
    std::vector<std::vector<char>> data_;
};

} /* end namespace datatools */
} /* end namespace megamol */
