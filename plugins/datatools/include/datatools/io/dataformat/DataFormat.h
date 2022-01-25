#pragma once

#include <filesystem>
#include <fstream>
#include <vector>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

struct AbstractFrame {
    virtual bool Read(std::ifstream& io) = 0;
    virtual bool Write(std::ofstream& io) = 0;
};

template<class Frame>
class AbstractDataFormat {
public:
    virtual ~AbstractDataFormat() = default;
    using FrameType = Frame;
    using FrameIndexType = typename FrameType::FrameIndexType;
    using FileType = std::filesystem::directory_entry;
    using FileListType = std::vector<FileType>;

    virtual std::unique_ptr<Frame> ReadFrame(std::ifstream& io, FrameIndexType idx) = 0;
    virtual void WriteFrame(std::ofstream& io, Frame const& frame) = 0;

    virtual FileListType EnumerateFramesInDirectory(FileType Path, std::string FilePattern) = 0;
};

template<class Format>
class AbstractDataContainer {
public:
    virtual ~AbstractDataContainer() = default;
    using FormatType = Format;
    using FrameType = typename Format::FrameType;
    using FrameIndexType = typename FrameType::FrameIndexType;

    std::unique_ptr<FrameType> ReadFrame(FrameIndexType idx) {
        // here we should actually grab the frames from some background thread that reads ahead and stuff?
        FrameType f;
        f.Read(IndexToIStream(idx));
        return std::make_unique<FrameType>(f);
    }

    void WriteFrame(FrameIndexType idx, FrameType const& frame) {};

    virtual std::ifstream IndexToIStream(FrameIndexType idx) = 0;
    virtual std::ofstream IndexToOStream(FrameIndexType idx) = 0;
};

// A directory containing several files, one for each frame
template<class Format>
class FolderContainer : public AbstractDataContainer<Format> {
    // this container supports arbitrary insertion and appending
public:
    using FrameType = typename Format::FrameType;
    using FrameIndexType = typename FrameType::FrameIndexType;

    bool Open(std::string location, std::string pattern) {
        files = Format::EnumerateFramesInDirectory(location, pattern);
        return true;
    }

    std::ifstream IndexToIStream(FrameIndexType idx) override {
        return std::ifstream (files[idx].path().string().c_str(), std::ifstream::binary);
    }
    std::ofstream IndexToOStream(FrameIndexType idx) override {
        // TODO some code for making paths when idx > what we already had
        // TODO what about sparse stuff, i.e. when the numbers were not consecutive?
        auto filename = files[idx].path().string().c_str();
        return std::ofstream(filename, std::ifstream::binary);
    }

private:
    typename Format::FileListType files;
};

// One big blob of data, each frame sitting at some offset
template<class Format>
class BlobContainer : public AbstractDataContainer<Format> {
    // TODO this container does not support frame insertion. it should support frame appending.
    bool Open(std::string location) {
        return true;
    }
};

// some other containers...?

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
