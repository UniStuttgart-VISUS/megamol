#pragma once

#include <filesystem>
#include <fstream>
#include <regex>
#include <vector>

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

struct AbstractFrame {
    virtual ~AbstractFrame() = default;
    virtual bool Read(std::ifstream& io) = 0;
    virtual bool Write(std::ofstream& io) = 0;
};

struct AbstractNaming {
    virtual ~AbstractNaming() = default;
    virtual std::regex Pattern() = 0;
};

template <class Frame>
struct BaseNumbering {
    virtual ~BaseNumbering() = default;
    using FrameIndexType = typename Frame::FrameIndexType;

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
    using FrameIndexType = typename FrameType::FrameIndexType;
    using FileType = std::filesystem::directory_entry;

    virtual std::unique_ptr<Frame> ReadFrame(std::ifstream& io, FrameIndexType idx) = 0;
    virtual void WriteFrame(std::ofstream& io, Frame const& frame) = 0;

};

// TODO read-ahead number
template<class Format>
class AbstractDataContainer {
public:
    virtual ~AbstractDataContainer() = default;
    using FormatType = Format;
    using FrameType = typename Format::FrameType;
    using FrameIndexType = typename FrameType::FrameIndexType;
    // TODO: Adrian's LRUCache here since we need most of the functionality anyway, so why not.
    using FrameCollection = std::unordered_map<FrameIndexType, FrameType>;

    // TODO generic EnumerateFrames that only files the frame indices with empty frames for now?

    std::unique_ptr<FrameType> ReadFrame(FrameIndexType idx) {
        // here we should actually grab the frames from some background thread that reads ahead and stuff?
        FrameType f;
        f.Read(IndexToIStream(idx));
        return std::make_unique<FrameType>(f);
    }

    void WriteFrame(FrameIndexType idx, FrameType const& frame) {}

    virtual std::ifstream IndexToIStream(FrameIndexType idx) = 0;
    virtual std::ofstream IndexToOStream(FrameIndexType idx) = 0;
};

// A directory containing several files, one for each frame
template<class Format>
class FolderContainer : public AbstractDataContainer<Format> {
    // this container supports arbitrary insertion and appending
public:
    using FileType = typename Format::FileType;
    using FileListType = std::vector<FileType>;
    using FrameType = typename Format::FrameType;
    using FrameIndexType = typename FrameType::FrameIndexType;

    FolderContainer(
        std::string location, std::unique_ptr<AbstractNaming> naming, BaseNumbering<FrameType> numbering = BaseNumbering<FrameType>())
            : naming(std::move(naming))
            , numbering(numbering) {
        files = EnumerateFramesInDirectory(FileType(location));
    }

    // TODO specific EnumerateFrames that uses the dir enumerator below?

    FileListType EnumerateFramesInDirectory(FileType Path) {
        auto r = std::regex(naming->Pattern());
        FileListType files;
        for (const auto& entry : std::filesystem::directory_iterator(Path)) {
            if (std::regex_match(entry.path().filename().string(), r)) {
                files.push_back(entry);
            }
        }
        std::sort(files.begin(), files.end());
        return files;
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
    FileListType files;
    BaseNumbering<FrameType> numbering;
    std::unique_ptr<AbstractNaming> naming;
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
