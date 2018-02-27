#ifndef MEGAMOL_DATATOOLS_PARTICLEVELOCITIESDIRANALYZER_H_INCLUDED
#define MEGAMOL_DATATOOLS_PARTICLEVELOCITIESDIRANALYZER_H_INCLUDED

#include <vector>

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/VolumeDataCall.h"

#include "vislib/math/Cuboid.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class ParticleVelocitiesDirAnalyzer : public megamol::core::Module {
public:
    /** Return module class name */
    static const char* ClassName(void) { return "ParticleVelocitiesDirAnalyzer"; }

    /** Return module class description */
    static const char* Description(void) {
        return "Computes a scalar field from velocity dir in user-defined direction.";
    }

    /** Module is always available */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    ParticleVelocitiesDirAnalyzer(void);

    /** Dtor */
    virtual ~ParticleVelocitiesDirAnalyzer(void);

protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);
private:
    bool getDataCallback(megamol::core::Call& c);

    bool getExtentCallback(megamol::core::Call& c);

    bool assertData(megamol::core::moldyn::VolumeDataCall* cvd);

    bool isDirty(void) const;

    void resetDirty(void);

    megamol::core::CalleeSlot dataOutSlot;

    megamol::core::CallerSlot dataInSlot;

    megamol::core::param::ParamSlot mainDirParamSlot;

    megamol::core::param::ParamSlot resParamSlot;

    int frameID;

    unsigned int frameCount;

    size_t in_datahash;

    size_t my_datahash;

    std::vector<float> volume;

    float cell_size[3];

    int cell_num[3];

    float stats[3];

    vislib::math::Cuboid<float> bbox;
}; /* end class ParticleVelocitiesDirAnalyzer */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_PARTICLEVELOCITIESDIRANALYZER_H_INCLUDED */