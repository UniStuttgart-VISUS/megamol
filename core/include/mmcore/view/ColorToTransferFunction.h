#ifndef MEGAMOLCORE_COLORTOTRANSFERFUNCTION_H_INCLUDED
#define MEGAMOLCORE_COLORTOTRANSFERFUNCTION_H_INCLUDED

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
 * Module that exposes a color parameter as transfer function.
 */
class ColorToTransferFunction : public megamol::core::Module, megamol::core::LuaInterpreter<ColorToTransferFunction> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static char const* ClassName(void) { return "ColorToTransferFunction"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static char const* Description(void) {
        return "Module that exposes a color parameter as transfer function";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ColorToTransferFunction(void);

    /** Dtor. */
    virtual ~ColorToTransferFunction(void);

protected:
    /** Lazy initialization */
    virtual bool create(void) override { return true; }

    /** Deferred destruction */
    virtual void release(void) override { }

private:
    bool getTFCallback(megamol::core::Call& c);

    bool colorUpdated(megamol::core::param::ParamSlot& p);

    int parseTF(lua_State* L);

    megamol::core::CalleeSlot getTFSlot;

    megamol::core::param::ParamSlot colorSlot;

    std::vector<float> tf;

    unsigned int texID;

    bool tfInvalidated = false;
}; /* end class ColorToTransferFunction */

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* end ifndef MEGAMOLCORE_COLORTOTRANSFERFUNCTION_H_INCLUDED */