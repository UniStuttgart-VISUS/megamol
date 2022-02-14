#pragma once

#include "DataFormat.h"
#include "ImageElementType.h"

namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

// do we want to pull out an interface? what kind of code do we write against this?
// which variants do we want to handle dynamically? different channel types probably. dimensionality is unlikely.
// different numbers of components?

// TODO: get rid of NumComponents. There is only one, keep multiple images if you want more channels.
// exception: RGBA8 as UINT32 ? Does it matter?

// typename ChannelType does not work, that is not a static decision.
// For some data sources you only know after actually opening the file what's in it.
// TODO: volume ist ein byte-vektor und wir benutzen span. fertig.
template<uint8_t Dimensions>
struct ImageFrame : AbstractFrame {
    static_assert(Dimensions > 0 && Dimensions < 4, "ImageFrame supports 1D, 2D, and 3D Images only");

    using SizeType = uint32_t;

    bool Read(std::istream& io) override {
        uint8_t dims, chanTypes;
        io.read(reinterpret_cast<char*>(&dims), sizeof(uint8_t));
        if (dims != Dimensions)
            throw std::invalid_argument("ImageFrame::Read: inappropriate number of dimensions in ImageFrame dump!");
        io.read(reinterpret_cast<char*>(&chanTypes), sizeof(uint8_t));
        elementType.Set(chanTypes);
        io.read(reinterpret_cast<char*>(&this->width), sizeof(SizeType));
        io.read(reinterpret_cast<char*>(&this->height), sizeof(SizeType));
        io.read(reinterpret_cast<char*>(&this->depth), sizeof(SizeType));
        data.resize(static_cast<std::size_t>(width) * height * depth);
        io.read(reinterpret_cast<char*>(this->data.data()), ByteSize());
        return true;
    }

    bool Write(std::ostream& io) const override {
        uint8_t const dims = Dimensions;
        io.write(reinterpret_cast<const char*>(&dims), sizeof(uint8_t));
        io.write(reinterpret_cast<const char*>(&elementType), sizeof(uint8_t));
        io.write(reinterpret_cast<const char*>(&this->width), sizeof(SizeType));
        io.write(reinterpret_cast<const char*>(&this->height), sizeof(SizeType));
        io.write(reinterpret_cast<const char*>(&this->depth), sizeof(SizeType));
        io.write(reinterpret_cast<const char*>(this->data.data()), ByteSize());
        return true;
    }

    void SetData(std::vector<uint8_t>&& data, ImageElementType channelType, SizeType width = 1, SizeType height = 1,
        SizeType depth = 1) {
        ASSERT(width * height * depth * channelType.ByteSize() == data.size());
        this->data = data;
        this->width = width;
        this->height = height;
        this->depth = depth;
        this->elementType = channelType;
    }

    // data will be controlled by this instance, caller loses access!
    void SetData(uint8_t* data, std::size_t count, ImageElementType channelType, SizeType width = 1, SizeType height = 1, SizeType depth = 1) {
        auto vec = std::vector(data, data + count);
        delete[] data;
        SetData(vec, channelType, width, height, depth);
    }

    template <typename T>
    constexpr T* ViewAs() const {
        return static_cast<const T*>(data.data());
    }

    template <typename ResultType>
    std::vector<ResultType> GetCopy() {
        std::vector<ResultType> out;
        out.reserve(NumElements());
        switch (elementType) {
        case ImageElementType::UINT8:
            CopyInto<ResultType, uint8_t>(out);
            break;
        case ImageElementType::UINT16:
            CopyInto<ResultType, uint16_t>(out);
            break;
        case ImageElementType::UINT32:
            // AKA case ChannelType::RGBA8:
            CopyInto<ResultType, uint32_t>(out);
            break;
        case ImageElementType::FLOAT:
            CopyInto<ResultType, float>(out);
            break;
        case ImageElementType::DOUBLE:
            CopyInto<ResultType, double>(out);
            break;
        default:
            throw std::logic_error("ImageFrame::GetCopy: invalid elementType");
        }
        return std::move(out);
    }

    template<typename ResultType>
    std::vector<ResultType> GetCopyNormalized() {
        std::vector<ResultType> out;
        out.reserve(NumElements());
        switch (elementType) {
        case ImageElementType::UINT8:
            CopyIntoNormalized<ResultType, uint8_t>(out);
            break;
        case ImageElementType::UINT16:
            CopyIntoNormalized<ResultType, uint16_t>(out);
            break;
        case ImageElementType::UINT32:
            // AKA case ChannelType::RGBA8:
            CopyIntoNormalized<ResultType, uint32_t>(out);
            break;
        case ImageElementType::FLOAT:
            CopyIntoNormalized<ResultType, float>(out);
            break;
        case ImageElementType::DOUBLE:
            CopyIntoNormalized<ResultType, double>(out);
            break;
        default:
            throw std::logic_error("ImageFrame::GetCopyNormalized: invalid elementType");
        }
        return std::move(out);
    }

    template<typename T>
    T GetValue(SizeType index) const {
        if (index >= NumElements())
            throw std::invalid_argument("ImageFrame::GetValue: index out of bounds");
        switch (elementType) {
        case ImageElementType::UINT8:
            return static_cast<T>(ViewAs<uint8_t>()[index]);
            break;
        case ImageElementType::UINT16:
            return static_cast<T>(ViewAs<uint16_t>()[index]);
            break;
        case ImageElementType::UINT32:
        // AKA case ChannelType::RGBA8:
            return static_cast<T>(ViewAs<uint32_t>()[index]);
            break;
        case ImageElementType::FLOAT:
            return static_cast<T>(ViewAs<float>()[index]);
            break;
        case ImageElementType::DOUBLE:
            return static_cast<T>(ViewAs<double>()[index]);
            break;
        default:
            throw std::logic_error("ImageFrame::GetValue: invalid elementType");
        }
    }

    template <typename T>
    void SetValue(SizeType index, T val) {
        if (index >= NumElements())
            throw std::invalid_argument("ImageFrame::SetValue: index out of bounds");
        switch (elementType) {
        case ImageElementType::UINT8:
            SetAbsolute<uint8_t>(index, val);
            break;
        case ImageElementType::UINT16:
            SetAbsolute<uint16_t>(index, val);
            break;
        case ImageElementType::UINT32:
        // AKA case ChannelType::RGBA8:
            SetAbsolute<uint32_t>(index, val);
            break;
        case ImageElementType::FLOAT:
            SetAbsolute<float>(index, val);
            break;
        case ImageElementType::DOUBLE:
            SetAbsolute<double>(index, val);
            break;
        default:
            throw std::logic_error("ImageFrame::SetValue: invalid elementType");
        }
    }

    template <typename T>
    void SetValueNormalized(SizeType index, T val, T maximum = std::numeric_limits<T>::max()) {
        if (index >= NumElements())
            throw std::invalid_argument("ImageFrame::SetValueNormalized: index out of bounds");
        const double relative = static_cast<double>(val) / maximum;
        switch (elementType) {
        case ImageElementType::UINT8:
            SetRelative<uint8_t>(index, relative);
            break;
        case ImageElementType::UINT16:
            SetRelative<uint16_t>(index, relative);
            break;
        case ImageElementType::UINT32:
            // AKA case ChannelType::RGBA8:
            SetRelative<uint32_t>(index, relative);
            break;
        case ImageElementType::FLOAT:
            SetRelative<float>(index, relative);
            break;
        case ImageElementType::DOUBLE:
            SetRelative<double>(index, relative);
            break;
        default:
            throw std::logic_error("ImageFrame::SetValueNormalized: invalid elementType");
        }
    }

    template<typename T>
    void SetValueNormalized2(SizeType index, T val, T maximum = std::numeric_limits<T>::max()) {
        if (index >= NumElements())
            throw std::invalid_argument("ImageFrame::SetValueNormalized: index out of bounds");
        const double relative = static_cast<double>(val) / maximum;
        Dispatcher<SetRelative_FW>::dispatch(elementType)(this, index, relative);
    }

    // https://stackoverflow.com/questions/16552166/c-function-dispatch-with-template-parameters
    template<typename FunctionWrapper>
    struct Dispatcher {
        static typename FunctionWrapper::Function* dispatch(ImageElementType::Value it) {
            switch (it) {
            case ImageElementType::UINT8:
                return &FunctionWrapper::template run<uint8_t>;
                break;
            case ImageElementType::UINT16:
                return &FunctionWrapper::template run<uint16_t>;
                break;
            case ImageElementType::UINT32:
                // AKA case ChannelType::RGBA8:
                return &FunctionWrapper::template run<uint32_t>;
                break;
            case ImageElementType::FLOAT:
                return &FunctionWrapper::template run<float>;
                break;
            case ImageElementType::DOUBLE:
                return &FunctionWrapper::template run<double>;
                break;
            default:
                throw std::logic_error("ImageFrame::Dispatcher: invalid elementType");
            }
        }
    };

    [[nodiscard]] SizeType ValueIndex(SizeType x, SizeType y = 0, SizeType z = 0) const {
        return (z * height + y) * width + x;
    }

    [[nodiscard]] std::size_t ByteSize() const override {
        return width * height * depth * elementType.ByteSize();
    }

    [[nodiscard]] std::size_t ElementSize() const {
        return elementType.ByteSize();
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
    template<typename T>
    T* AccessAs() {
        return reinterpret_cast<T*>(data.data());
    }

    template<typename T>
    void SetRelative(SizeType index, double relative) {
        AccessAs<T>()[index] = static_cast<T>(relative * std::numeric_limits<T>::max());
    }

    struct SetRelative_FW {
        using Function = void(ImageFrame<Dimensions>*, SizeType, double);

        template<typename T>
        static void run(ImageFrame<Dimensions>* that, SizeType index, double relative) {
            that->SetRelative<T>(index, relative);
        };
    };

    template<typename Dest, typename Source>
    void SetAbsolute(SizeType index, Source val) {
        AccessAs<Dest>()[index] = static_cast<Dest>(val);
    }

    template<typename Dest, typename Source>
    void CopyInto(std::vector<Dest>& out) {
        std::transform(ViewAs<Source>(), ViewAs<Source>() + NumElements(), std::back_inserter(out),
            [](const auto& val) { return static_cast<Dest>(val); });
    }

    template<typename Dest, typename Source>
    void CopyIntoNormalized(std::vector<Dest>& out) {
        std::transform(ViewAs<Source>(), ViewAs<Source>() + NumElements(), std::back_inserter(out), [](const auto& val) {
                return static_cast<Dest>((val / std::numeric_limits<Source>::max()) * std::numeric_limits<Dest>::max());
            });
    }

    std::vector<uint8_t> data;
    SizeType width = 0, height = 0, depth = 0;
    ImageElementType elementType = ImageElementType::UINT8;
};

//using Uint8Image2DFrame = ImageFrame<2>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
