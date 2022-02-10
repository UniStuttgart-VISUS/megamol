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

struct CSVFrame : AbstractFrame {
    using FrameIndexType = uint32_t;
    using DataIndexType = uint32_t;

    FrameIndexType FrameIndex = 0;
    DataIndexType NumColumns = 0;
    DataIndexType NumRows = 0;
    std::vector<CSVColumnInfo> ColumnInfos;
    std::vector<float> Values;

    //bool Read(std::ifstream& io) override {
    //    return true;
    //}
    //bool Write(std::ofstream& io) override {
    //    return true;
    //}
};

using CSVDataFormat = AbstractDataFormat<CSVFrame>;
using CSVFileCollection = FolderContainer<CSVDataFormat>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
