#pragma once

#include "DataFormat.h"

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

struct CSVColumnInfo {
    std::string name;
    // TODO
};

struct CSVFrame {
    using FrameIndexType = uint32_t;
    using DataIndexType = uint32_t;

    FrameIndexType FrameIndex;
    DataIndexType NumColumns;
    DataIndexType NumRows;
    std::vector<CSVColumnInfo> ColumnInfos;
    std::vector<float> Values;
};

class CSVDataFormat : public AbstractDataFormat<CSVFrame> {
    std::unique_ptr<CSVFrame> ReadFrame(std::ifstream& io) override {
        return std::make_unique<CSVFrame>();
    }
    void WriteFrame(std::ofstream& io, CSVFrame frame) override{}
};

using CSVFileCollection = FolderContainer<CSVDataFormat>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
