#pragma once

#include "datatools/AbstractParticleManipulator.h"

namespace megamol {
namespace datatools {

class IColToIdentity : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName(void) {
        return "IColToIdentity";
    }

    /** Return module class description */
    static const char* Description(void) {
        return "Copies the ICol stream into the Identity stream of MPDC";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    IColToIdentity(void);

    /** Dtor */
    ~IColToIdentity(void) override;

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
    std::vector<char> ids;
};

} /* end namespace datatools */
} /* end namespace megamol */
