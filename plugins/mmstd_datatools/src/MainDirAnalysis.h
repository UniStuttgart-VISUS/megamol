#ifndef MEGAMOL_DATATOOLS_MAINDIRANALYSIS_H_INCLUDED
#define MEGAMOL_DATATOOLS_MAINDIRANALYSIS_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

class MainDirAnalysis : public megamol::core::Module {
public:
    /** Return module class name */
    static char const* ClassName(void) {
        return "MainDirAnalysis";
    }

    /** Return module class description */
    static char const* Description(void) {
        return "Exports magnitude of velocity in one general direction.";
    }

    /** Module is always available */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    MainDirAnalysis(void);

    /** Dtor */
    virtual ~MainDirAnalysis(void);
protected:
    /** Lazy initialization of the module */
    virtual bool create(void);

    /** Resource release */
    virtual void release(void);
private:
    bool getDataCallback(megamol::core::Call& c);

    bool getExtentCallback(megamol::core::Call& c);

    megamol::core::CalleeSlot dataOutSlot;

    megamol::core::CallerSlot dataInSlot;
}; /* end class MainDirAnalysis */

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOL_DATATOOLS_MAINDIRANALYSIS_H_INCLUDED */