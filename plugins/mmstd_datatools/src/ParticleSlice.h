#ifndef MEGAMOL_DATATOOLS_PARTICLESLICE_H_INCLUDED
#define MEGAMOL_DATATOOLS_PARTICLESLICE_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class ParticleSlice : public megamol::core::Module {
public:
    /** Return module class name */
    static char const* ClassName(void) {
        return "ParticleSlice";
    }

    /** Return module class description */
    static char const* Description(void) {
        return "Module to which exports only a slice of a dataset";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    ParticleSlice(void);

    /** Dtor */
    virtual ~ParticleSlice(void);
protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);
private:
    bool getDataCallback(megamol::core::Call& c);

    bool getExtentCallback(megamol::core::Call& c);

    bool assertData();

    megamol::core::CalleeSlot dataOutSlot;

    megamol::core::CallerSlot dataInSlot;

    megamol::core::param::ParamSlot axisSlot;

    megamol::core::param::ParamSlot thicknessSlot;

    megamol::core::param::ParamSlot positionSlot;

    size_t datahash;

    unsigned int frameID;

    std::vector<std::vector<float>> positions;

    std::vector<std::vector<float>> colors;

    std::vector<std::vector<float>> directions;
}; /* end class ParticleSlice */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_PARTICLESLICE_H_INCLUDED */