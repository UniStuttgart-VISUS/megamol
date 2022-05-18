#include "OpenGL_Context.h"

#include <algorithm>
#include <iterator>
#include <sstream>


bool megamol::frontend_resources::OpenGL_Context::isVersionGEQ(int major, int minor) const {
    if (major_ < major) {
        return false;
    }
    if (major_ > major) {
        return true;
    }
    if (minor_ >= minor) {
        return true;
    }
    return false;
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
