#ifndef PBS_PBSDATACALL_H_INCLUDED
#define PBS_PBSDATACALL_H_INCLUDED

#include "pbs.h"
#include "mmcore/AbstractGetData3DCall.h"

namespace megamol {
namespace pbs {

class PBS_API PBSDataCall : public core::AbstractGetData3DCall {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char *ClassName(void) {
        return "PBSDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char *Description(void) {
        return "Call to transport PBS data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return its name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    /** ctor */
    PBSDataCall(void);

    /** dtor */
    virtual ~PBSDataCall(void);
protected:
private:
}; /* end class PBSDataCall */

} /* end namespace pbs */
} /* end namespace megamol */

#endif /* end ifndef PBS_PBSDATACALL_H_INCLUDED */