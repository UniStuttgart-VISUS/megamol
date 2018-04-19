#ifndef MEGAMOLCORE_TRANSFERFUNCTION1D_H_INCLUDED
#define MEGAMOLCORE_TRANSFERFUNCTION1D_H_INCLUDED

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "mmcore/view/CallGetTransferFunction.h"

#include "mmcore/LuaInterpreter.h"

namespace megamol {
namespace core {
namespace view {

/**
 * Module defining a linear transfer function which can be manipulated by the configurator.
 */
class TransferFunction1D : public megamol::core::Module, megamol::core::LuaInterpreter<TransferFunction1D> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static char const* ClassName(void) { return "TransferFunction1D"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static char const* Description(void) {
        return "Module exposing linear transfer function defined by the configurator";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    TransferFunction1D(void);

    /** Dtor. */
    virtual ~TransferFunction1D(void);

protected:
    /** Lazy initialization */
    virtual bool create(void);

    /** Deferred destruction */
    virtual void release(void);

private:
    bool getTFCallback(megamol::core::Call& c);

    bool tfUpdated(megamol::core::param::ParamSlot& p);

    int parseTF(lua_State* L);

    //LuaInterpreter<TransferFunction1D> lua;

    megamol::core::CalleeSlot getTFSlot;

    megamol::core::param::ParamSlot tfParamSlot;

    std::vector<float> tf;

    unsigned int texID;

    bool tfChanged = false;
}; /* end class TransferFunction1D */

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_TRANSFERFUNCTION1D_H_INCLUDED */