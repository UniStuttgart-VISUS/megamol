#pragma once

#include "DataFormat.h"
#include "mmcore/utility/graphics/ScreenShotComments.h"

#include <regex>
#include <algorithm>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

// this this goes into the call. with that exact read/write signature for brain dumps
struct ImageFrame : AbstractFrame {
    using SizeType = uint32_t;

    struct ChannelType {
        enum Value : uint8_t {UINT8, FLOAT};
        ChannelType() = default;
        constexpr ChannelType(Value v) : val(v) {}
        constexpr operator Value() const {
            return val;
        }
        explicit operator bool() = delete;
        constexpr uint8_t GetByteSize() const {
            switch (val) {
            case UINT8:
                return sizeof(uint8_t);
            case FLOAT:
                return sizeof(float);
            }
            return sizeof(uint8_t);
        }
    private:
        Value val;
    };

    ChannelType Type;
    uint8_t NumChannels = 1;
    SizeType Width = 0, Height = 0;

    std::vector<uint8_t> Values;

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

// todo where would we point when we read a mmpld frame?
// todo: end pointer?
// does this map to MMPLD and ADIOS?
//using PNGDataFormat = AbstractDataFormat<PNGFrame>;
//using PNGFileCollection = FolderContainer<PNGDataFormat>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
