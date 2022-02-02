#pragma once

#include <filesystem>
#include <fstream>
#include <regex>
#include <vector>

#include "LRUCache.h"

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

// every call ideally should implement this, as a generic interface for "brain dumps"
// this has *nothing* to do with proper file formats. readers support these to provide data via the said call
struct AbstractFrame {
    virtual ~AbstractFrame() = default;
    virtual bool Read(std::istream& io) = 0;
    virtual bool Write(std::ostream& io) const = 0;
    [[nodiscard]] virtual std::size_t ByteSize() const = 0;
};

struct AbstractMetadata {
    using FrameIndexType = uint32_t;
};

struct AbstractNaming {
    virtual ~AbstractNaming() = default;
    virtual std::regex Pattern() = 0;
};

template<typename FrameIndexType>
struct BaseNumbering {
    virtual ~BaseNumbering() = default;

    BaseNumbering(uint8_t digits = 5) : digits(digits) {
        reg = std::regex(std::string("(\d{") + std::to_string(digits) + "})");
    }

    virtual FrameIndexType ExtractNumber(std::string filename) {
        std::smatch matches;
        if (std::regex_match(filename, matches, reg)) {
            return std::stoi(matches[0].str());
        }
        return static_cast<FrameIndexType>(0);
    }

    virtual std::string MakeFileName(std::string prefix, FrameIndexType idx) {
        std::stringstream str;
        str << std::setfill('0') << std::setw(digits) << idx;
        return prefix + str.str();
    }

private:
    std::regex reg;
    uint8_t digits;
};

template<class Frame>
class AbstractDataFormat {
public:
    virtual ~AbstractDataFormat() = default;
    using FrameType = Frame;
    using FileType = std::filesystem::directory_entry;

    //virtual std::unique_ptr<Frame> ReadFrame(std::ifstream& io, FrameIndexType idx) = 0;
    //virtual void WriteFrame(std::ofstream& io, Frame const& frame) = 0;
};

// TODO read-ahead number
template<class Format>
class AbstractDataContainer {
public:
    using FormatType = Format;
    using FrameType = typename Format::FrameType;
    using FrameIndexType = typename FrameType::FrameIndexType;
    using FrameCollection = LRUCache<AbstractDataContainer>;

    AbstractDataContainer(FrameIndexType readAhead = 3)
            : readAhead(readAhead)
            , frames(LRUCache<AbstractDataContainer>()) {
    }

    virtual ~AbstractDataContainer() = default;

    // TODO generic EnumerateFrames that only files the frame indices with empty frames for now?
    virtual FrameIndexType EnumerateFrames() = 0;

    std::shared_ptr<FrameType> ReadFrame(FrameIndexType idx) {
        // here we should actually grab the frames from some background thread that reads ahead and stuff?
        return frames.findOrCreate(idx, *this);
    }

    void WriteFrame(FrameIndexType idx, FrameType const& frame) {}

    virtual std::ifstream IndexToIStream(FrameIndexType idx) = 0;
    virtual std::ofstream IndexToOStream(FrameIndexType idx) = 0;
protected:
    FrameIndexType readAhead;
    FrameCollection frames;
};

// A directory containing several files, one for each frame
template<class Format>
class FolderContainer : public AbstractDataContainer<Format> {
    // this container supports arbitrary insertion and appending
public:
    using FileType = typename Format::FileType;
    using FileListType = std::vector<FileType>;
    using FrameIndexType = typename Format::FrameIndexType;

    // TODO fix basenumbering

    FolderContainer(std::string location, std::unique_ptr<AbstractNaming> naming,
        std::unique_ptr<BaseNumbering<FrameIndexType>> numbering = std::make_unique<BaseNumbering<FrameIndexType>>())
            : files(EnumerateFramesInDirectory(FileType(location)))
            , naming(std::move(naming))
            , numbering(std::move(numbering)) {
    }

    FrameIndexType EnumerateFrames() override {
        return static_cast<FrameIndexType>(files.size());
    }

    FileListType EnumerateFramesInDirectory(FileType Path) {
        auto r = std::regex(naming->Pattern());
        FileListType fileList;
        for (const auto& entry : std::filesystem::directory_iterator(Path)) {
            if (std::regex_match(entry.path().filename().string(), r)) {
                fileList.push_back(entry);
            }
        }
        std::sort(fileList.begin(), fileList.end());
        return fileList;
    }

    std::ifstream IndexToIStream(FrameIndexType idx) override {
        return std::ifstream(files[idx].path().string().c_str(), std::ifstream::binary);
    }
    std::ofstream IndexToOStream(FrameIndexType idx) override {
        // TODO some code for making paths when idx > what we already had
        // TODO what about sparse stuff, i.e. when the numbers were not consecutive?
        // TODO the clang dangling pointer
        auto filename = files[idx].path().string().c_str();
        return std::ofstream(filename, std::ifstream::binary);
    }

private:
    FileListType files;
    std::unique_ptr<AbstractNaming> naming;
    std::unique_ptr<BaseNumbering<FrameIndexType>> numbering;
};

// One big blob of data, each frame sitting at some offset
template<class Format>
class BlobContainer : public AbstractDataContainer<Format> {
    // TODO this container does not support frame insertion. it should support frame appending.
    // totally TODO actually
    bool Open(std::string location) {
        return true;
    }
};

// some other containers...?

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
