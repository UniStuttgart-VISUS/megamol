#pragma once

#include "DataFormat.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"

#include <regex>
#include <algorithm>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

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

class PNGDataFormat : public AbstractDataFormat<PNGFrame> {
    std::unique_ptr<PNGFrame> ReadFrame(std::ifstream& io, PNGFrame::FrameIndexType idx) override {
        return std::make_unique<PNGFrame>();
    }

    void WriteFrame(std::ofstream& io, PNGFrame const& frame) override {}

    FileListType EnumerateFramesInDirectory(FileType Path, std::string FilePattern) override {
        // TODO how to separate name, extension, and frame number?
        auto r = std::regex(FilePattern);
        FileListType files;
        for (const auto& entry : std::filesystem::directory_iterator(Path)) {
            if (std::regex_match(entry.path().filename().string(), r)) {
                files.push_back(entry);
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }
};

using PNGFileCollection = FolderContainer<PNGDataFormat>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
