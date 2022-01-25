#pragma once

#include "DataFormat.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"

#include <regex>
#include <algorithm>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

// todo where would we point when we read a mmpld frame?
// todo: end pointer?
// does this map to MMPLD and ADIOS?
struct PNGFrame : AbstractFrame {
    using FrameIndexType = uint32_t;
    using SizeType = uint32_t;

    FrameIndexType FrameIndex = 0;
    SizeType Width = 0, Height = 0;
    megamol::core::utility::graphics::ScreenShotComments::comments_storage_map comments;

    std::vector<float> Values;

    bool Read(std::ifstream& io) override {
        return true;
    }
    bool Write(std::ofstream& io) override {
        return true;
    }
};

struct PNGNaming : AbstractNaming {
    std::regex Pattern() override {
        return std::regex("^.*?\\.png");
    }
};

using PNGDataFormat = AbstractDataFormat<PNGFrame>;
using PNGFileCollection = FolderContainer<PNGDataFormat>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
