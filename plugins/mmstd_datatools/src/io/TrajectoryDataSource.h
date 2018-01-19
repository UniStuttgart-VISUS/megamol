#ifndef MEGAMOL_STDPLUGIN_DATATOOLS_TRAJECTORYDATASOURCE_H_INCLUDED
#define MEGAMOL_STDPLUGIN_DATATOOLS_TRAJECTORYDATASOURCE_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace io {

class TrajectoryDataSource : public megamol::core::Module {
public:
    /** Return module class name */
    static char const* ClassName(void) {
        return "TrajectoryDataSource";
    }

    /** Return module class description */
    static char const* Description(void) {
        return "Reads a trajectory file and exposes its contents as line data.";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    TrajectoryDataSource(void);

    /** Dtor */
    virtual ~TrajectoryDataSource(void);
protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);
private:
    bool getDataCallback(megamol::core::Call& c);

    bool getExtentCallback(megamol::core::Call& c);

    bool assertData();

    megamol::core::CalleeSlot trajOutSlot;

    megamol::core::param::ParamSlot trajFilepath;
}; /* end class TrajectoryDataSource */

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_STDPLUGIN_DATATOOLS_TRAJECTORYDATASOURCE_H_INCLUDED */
