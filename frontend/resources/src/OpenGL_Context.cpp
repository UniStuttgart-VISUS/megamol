#include "OpenGL_Context.h"

#include <algorithm>
#include <iterator>
#include <sstream>

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


bool megamol::frontend_resources::OpenGL_Context::areExtAvailable(std::string const& exts) const {
    std::istringstream iss(exts);
    bool avail = true;
    std::for_each(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
        [&avail, this](auto const& ext) { avail = avail && isExtAvailable(ext); });
    return avail;
}
