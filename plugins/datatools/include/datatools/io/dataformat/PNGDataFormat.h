#pragma once

#include "DataFormat.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"

#include <regex>
#include <algorithm>
#include <stdexcept>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

struct PNGNaming : AbstractNaming {
    std::regex Pattern() override {
        return std::regex("^.*?\\.png");
    }
};

// todo where would we point when we read a mmpld frame?
// todo: end pointer?
// does this map to MMPLD and ADIOS?
//using PNGDataFormat = AbstractDataFormat<PNGFrame>;
//using PNGFileCollection = FolderContainer<PNGDataFormat>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
