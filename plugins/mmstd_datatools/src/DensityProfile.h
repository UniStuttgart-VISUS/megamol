#ifndef MEGAMOL_DATATOOLS_DENSITYPROFILE_H_INCLUDED
#define MEGAMOL_DATATOOLS_DENSITYPROFILE_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class DensityProfile : public megamol::core::Module {
public:
    /** Return module class name */
    static char const* ClassName(void) {
        return "DensityProfile";
    }

    /** Return module class description */
    static char const* Description(void) {
        return "Module to compute the density profile of a MD dataset along its longest extent";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    DensityProfile(void);

    /** Dtor */
    virtual ~DensityProfile(void);
protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);
private:
    bool getDataCallback(megamol::core::Call& c);

    bool getHashCallback(megamol::core::Call& c);

    bool assertData(void);

    megamol::core::CalleeSlot outDataSlot;

    megamol::core::CallerSlot inDataSlot;

    megamol::core::param::ParamSlot sliceSizeFactorSlot;

    size_t datahash;

    std::vector<float> data_;

    std::pair<float, float> data_minmax_;

    megamol::stdplugin::datatools::floattable::CallFloatTableData::ColumnInfo data_ci_;
}; /* end class DensityProfile */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_DENSITYPROFILE_H_INCLUDED */