#pragma once

#include <fstream>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

struct AbstractFrame {
    virtual bool Read(std::ifstream& io) = 0;
    virtual bool Write(std::ofstream& io) = 0;
};

template<class F>
class AbstractDataFormat {
public:
    virtual ~AbstractDataFormat() = default;
    using FrameType = F;
    virtual std::unique_ptr<F> ReadFrame(std::ifstream& io, typename F::FrameIndexType idx) = 0;
    virtual void WriteFrame(std::ofstream& io, F const& frame) = 0;
};

template<class F>
class AbstractDataContainer {
public:
    virtual ~AbstractDataContainer() = default;
    using Format = F;
    using FrameIndex = typename F::FrameType::FrameIndexType;

    virtual bool Open(std::string location) = 0; // todo mode
};

// A directory containing several files, one for each frame
template<class F>
class FolderContainer : public AbstractDataContainer<F> {
    bool Open(std::string location) override {
        return true;
    }
};

// One big blob of data, each frame sitting at some offset
template<class F>
class BlobContainer : public AbstractDataContainer<F> {
    bool Open(std::string location) override {
        return true;
    }
};

// some other containers...?

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
