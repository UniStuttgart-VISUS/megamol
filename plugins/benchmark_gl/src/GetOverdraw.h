#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "glowl/FramebufferObject.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/BufferObject.hpp"

namespace megamol::benchmark_gl {
class GetOverdraw : public core::Module {
public:
    /**
     * Gets the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "GetOverdraw";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "GetOverdraw";
    }

    /**
     * Gets whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    GetOverdraw(void);

    /** Dtor. */
    virtual ~GetOverdraw(void);

protected:
    bool create() override;

    void release() override;

private:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core::Call& c);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core::Call& c);

    bool passthrough(core::Call& c);

    bool get_tex(core::Call& c);

    bool get_tex_meta(core::Call& c);

    std::shared_ptr<glowl::FramebufferObject> rt_;

    std::shared_ptr<glowl::Texture2D> stencil_buffer_copy_;

    std::shared_ptr<glowl::BufferObject> pbo_;

    core::CalleeSlot get_tex_slot_;

    core::CalleeSlot render_out_slot_;

    core::CallerSlot render_in_slot_;

    uint64_t version_ = 0;
};
} // namespace megamol::benchmark_gl
