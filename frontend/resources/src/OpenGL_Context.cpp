#include "OpenGL_Context.h"

#include "glad/gl.h"


bool megamol::frontend_resources::OpenGL_Context::isVersionGEQ(int major, int minor) const {
    if (version_ < GLAD_MAKE_VERSION(major, minor)) {
        return false;
    }
    return true;
}


bool megamol::frontend_resources::OpenGL_Context::isExtAvailable(std::string const& ext) const {
    auto const fit = std::find(ext_.begin(), ext_.end(), ext);
    return fit != ext_.end();
}
