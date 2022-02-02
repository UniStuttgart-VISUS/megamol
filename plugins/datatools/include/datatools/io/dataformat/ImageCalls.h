#pragma once

#include "DataFormat.h"

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

// do we want to pull out an interface? what kind of code do we write against this?
// which variants do we want to handle dynamically? different channel types probably. dimensionality is unlikely.
// different numbers of components?
template<typename ChannelType, uint8_t Dimensions>
struct ImageFrame : AbstractFrame {
    static_assert(Dimensions > 0 && Dimensions < 4, "ImageFrame supports 1D, 2D, and 3D Images only");

    using SizeType = uint32_t;
    using ComponentSizeType = uint8_t;

    bool Read(std::istream& io) override {
        uint8_t dims;
        io.read(reinterpret_cast<char*>(&dims), sizeof(uint8_t));
        if (dims != Dimensions)
            throw std::invalid_argument("inappropriate number of dimensions in ImageFrame dump!");
        io.read(static_cast<char*>(&this->numComponents), sizeof(ComponentSizeType));
        io.read(static_cast<char*>(&this->width), sizeof(SizeType));
        io.read(static_cast<char*>(&this->height), sizeof(SizeType));
        io.read(static_cast<char*>(&this->depth), sizeof(SizeType));
        data.resize(width * height * depth * numComponents);
        io.read(this->data.data(), ByteSize());
        return true;
    }

    bool Write(std::ostream& io) const override {
        uint8_t dims = Dimensions;
        io.write(reinterpret_cast<char*>(&dims), sizeof(uint8_t));
        io.write(static_cast<char*>(&this->numComponents), sizeof(ComponentSizeType));
        io.write(static_cast<char*>(&this->width), sizeof(SizeType));
        io.write(static_cast<char*>(&this->height), sizeof(SizeType));
        io.write(static_cast<char*>(&this->depth), sizeof(SizeType));
        io.write(this->data.data(), ByteSize());
        return true;
    }

    void SetData(std::vector<ChannelType>&& data, ComponentSizeType numComponents = 1, SizeType width = 1,
        SizeType height = 1, SizeType depth = 1) {
        ASSERT(width * height * depth * numComponents == data.size());
        // some asserts regarding Dimensions and parameters? or rather not?
        this->data = data;
        this->numComponents = numComponents;
        this->width = width;
        this->height = height;
        this->depth = depth;
    }

    const std::vector<ChannelType> GetData() {
        return data;
    }

    ChannelType GetValue(SizeType index) {
        return data[index];
    }

    void SetValue(SizeType index, ChannelType val) {
        data[index] = val;
    }

    inline SizeType ValueIndex(SizeType x, SizeType y = 0, SizeType z = 0, ComponentSizeType c = 0) {
        return ((z * height + y) * width + x) * numComponents + c;
    }

    [[nodiscard]] std::size_t ByteSize() const override {
        return width * height * depth * numComponents * sizeof(ChannelType);
    }

    [[nodiscard]] std::size_t ElementSize() const {
        return numComponents * sizeof(ChannelType);
    }

    [[nodiscard]] SizeType NumElements() const {
        return width * height * depth;
    }

    [[nodiscard]] SizeType Width() const {
        return width;
    }

    [[nodiscard]] SizeType Height() const {
        return height;
    }

    [[nodiscard]] SizeType Depth() const {
        return depth;
    }

private:
    std::vector<ChannelType> data;
    ComponentSizeType numComponents = 1;
    SizeType width = 0, height = 0, depth = 0;
};

using Uint8Image2DFrame = ImageFrame<uint8_t, 2>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
