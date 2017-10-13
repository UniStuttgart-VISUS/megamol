#ifndef PBS_PBSREADER_H_INCLUDED
#define PBS_PBSREADER_H_INCLUDED

#include "mmcore/Module.h"

namespace megamol {
namespace pbs {

/** Module to read ZFP compressed point dumps. */
class PBSDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "PBSDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Data reader module for ZFP compressed point dumps files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    PBSDataSource(void);

    /** dtor */
    virtual ~PBSDataSource(void);
protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);
private:
};

} /* end namespace pbs */
} /* end namespace megamol */

#endif // end ifndef PBS_PBSREADER_H_INCLUDED
