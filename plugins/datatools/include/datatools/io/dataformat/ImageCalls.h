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
struct ImageChannel {
    static_assert(Dimensions > 0 && Dimensions < 4, "ImageChannel supports 1D, 2D, and 3D Images only");

    using SizeType = uint32_t;

    void Read(std::istream& io) {
        uint8_t dims, chanTypes;
        io.read(reinterpret_cast<char*>(&dims), sizeof(uint8_t));
        if (dims != Dimensions)
            throw std::invalid_argument("ImageChannel::Read: inappropriate number of dimensions in ImageChannel dump!");
        io.read(reinterpret_cast<char*>(&chanTypes), sizeof(uint8_t));
        elementType.Set(chanTypes);
        io.read(reinterpret_cast<char*>(&this->width), sizeof(SizeType));
        io.read(reinterpret_cast<char*>(&this->height), sizeof(SizeType));
        io.read(reinterpret_cast<char*>(&this->depth), sizeof(SizeType));
        data.resize(static_cast<std::size_t>(width) * height * depth);
        io.read(reinterpret_cast<char*>(this->data.data()), ByteSize());
    }

    void Write(std::ostream& io) const {
        uint8_t const dims = Dimensions;
        io.write(reinterpret_cast<const char*>(&dims), sizeof(uint8_t));
        io.write(reinterpret_cast<const char*>(&elementType), sizeof(uint8_t));
        io.write(reinterpret_cast<const char*>(&this->width), sizeof(SizeType));
        io.write(reinterpret_cast<const char*>(&this->height), sizeof(SizeType));
        io.write(reinterpret_cast<const char*>(&this->depth), sizeof(SizeType));
        io.write(reinterpret_cast<const char*>(this->data.data()), ByteSize());
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
    void SetData(uint8_t* data, std::size_t count, ImageElementType channelType, SizeType width = 1,
        SizeType height = 1, SizeType depth = 1) {
        auto vec = std::vector(data, data + count);
        delete[] data;
        SetData(vec, channelType, width, height, depth);
    }

    template<typename T>
    constexpr const T* ViewAs() const {
        return reinterpret_cast<const T*>(data.data());
    }

    template<typename ReturnType>
    std::vector<ReturnType> GetCopy() const {
        std::vector<ReturnType> out;
        out.reserve(NumElements());
        Dispatcher<GetCopy_FW<ReturnType>>::dispatch(elementType)(this, out);
        return out;
    }

    template<typename ReturnType>
    std::vector<ReturnType> GetCopyNormalized(ReturnType maximum = std::numeric_limits<ReturnType>::max()) const {
        std::vector<ReturnType> out;
        out.reserve(NumElements());
        Dispatcher<GetCopyNormalized_FW<ReturnType>>::dispatch(elementType)(this, out, maximum);
        return out;
    }

    template<typename ReturnType>
    ReturnType GetValue(SizeType index) const {
        if (index >= NumElements())
            throw std::invalid_argument("ImageChannel::GetValue: index out of bounds");
        return Dispatcher<GetAbsolute_FW<ReturnType>>::dispatch(elementType)(this, index);
    }

    // this still feels wonky
    template<typename T>
    T* begin() {
        if (typeid(T) != elementType.TypeId()) {
            throw std::invalid_argument("cannot generate iterator for a type different from the contents");
        }
        return &AccessAs<T>()[0];
    }

    template<typename T>
    T* end() {
        if (typeid(T) != elementType.TypeId()) {
            throw std::invalid_argument("cannot generate iterator for a type different from the contents");
        }
        return &AccessAs<T>()[data.size()];
    }

    template<typename ReturnType>
    ReturnType GetValueNormalized(SizeType index, ReturnType maximum = std::numeric_limits<ReturnType>::max()) const {
        if (index >= NumElements())
            throw std::invalid_argument("ImageChannel::GetValueNormalized: index out of bounds");
        return Dispatcher<GetRelative_FW<ReturnType>>::dispatch(elementType)(this, index, maximum);
    }

    template<typename InputType>
    void SetValue(SizeType index, InputType val) {
        if (index >= NumElements())
            throw std::invalid_argument("ImageChannel::SetValue: index out of bounds");
        Dispatcher<SetAbsolute_FW<InputType>>::dispatch(elementType)(this, index, val);
    }

    template<typename T>
    void SetValueNormalized(SizeType index, T val, T maximum = std::numeric_limits<T>::max()) {
        if (index >= NumElements())
            throw std::invalid_argument("ImageChannel::SetValueNormalized: index out of bounds");
        const double relative = static_cast<double>(val) / maximum;
        Dispatcher<SetRelative_FW>::dispatch(elementType)(this, index, relative);
    }

    [[nodiscard]] SizeType ValueIndex(SizeType x, SizeType y = 0, SizeType z = 0) const {
        return (z * height + y) * width + x;
    }

    [[nodiscard]] std::size_t ByteSize() const {
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
            case ImageElementType::RGBA8:
                return &FunctionWrapper::template run<uint32_t>;
                break;
            case ImageElementType::FLOAT:
                return &FunctionWrapper::template run<float>;
                break;
            case ImageElementType::DOUBLE:
                return &FunctionWrapper::template run<double>;
                break;
            default:
                throw std::logic_error("ImageChannel::Dispatcher: invalid elementType");
            }
        }
    };

    template<typename T>
    T* AccessAs() {
        return reinterpret_cast<T*>(data.data());
    }

    template<typename T>
    void SetRelative(SizeType index, double relative) {
        AccessAs<T>()[index] = static_cast<T>(relative * std::numeric_limits<T>::max());
    }

    template<typename ReturnType>
    struct GetAbsolute_FW {
        using Function = ReturnType(ImageChannel<Dimensions> const*, SizeType);

        template<typename T>
        static ReturnType run(ImageChannel<Dimensions> const* that, SizeType index) {
            return static_cast<ReturnType>(that->ViewAs<T>()[index]);
        }
    };

    template<typename ReturnType>
    struct GetRelative_FW {
        using Function = ReturnType(ImageChannel<Dimensions> const*, SizeType, ReturnType);

        template<typename T>
        static ReturnType run(ImageChannel<Dimensions> const* that, SizeType index, ReturnType maximum) {
            return static_cast<ReturnType>(
                (that->ViewAs<T>()[index] / static_cast<double>(std::numeric_limits<T>::max())) * maximum);
        }
    };

    template<typename InputType>
    struct SetAbsolute_FW {
        using Function = void(ImageChannel<Dimensions>*, SizeType, InputType);

        template<typename T>
        static void run(ImageChannel<Dimensions>* that, SizeType index, InputType val) {
            that->SetAbsolute<T, InputType>(index, val);
        }
    };

    struct SetRelative_FW {
        using Function = void(ImageChannel<Dimensions>*, SizeType, double);

        template<typename T>
        static void run(ImageChannel<Dimensions>* that, SizeType index, double relative) {
            that->SetRelative<T>(index, relative);
        }
    };

    template<typename ResultType>
    struct GetCopy_FW {
        using Function = void(ImageChannel<Dimensions> const*, std::vector<ResultType>&);

        template<typename T>
        static void run(ImageChannel<Dimensions> const* that, std::vector<ResultType>& out) {
            that->CopyInto<ResultType, T>(out);
        }
    };

    template<typename ResultType>
    struct GetCopyNormalized_FW {
        using Function = void(ImageChannel<Dimensions> const*, std::vector<ResultType>&, ResultType);

        template<typename T>
        static void run(ImageChannel<Dimensions> const* that, std::vector<ResultType>& out, ResultType maximum) {
            that->CopyIntoNormalized<ResultType, T>(out, maximum);
        }
    };

    template<typename Dest, typename Source>
    void SetAbsolute(SizeType index, Source val) {
        AccessAs<Dest>()[index] = static_cast<Dest>(val);
    }

    template<typename Dest, typename Source>
    void CopyInto(std::vector<Dest>& out) const {
        std::transform(ViewAs<Source>(), ViewAs<Source>() + NumElements(), std::back_inserter(out),
            [](const auto& val) { return static_cast<Dest>(val); });
    }

    template<typename Dest, typename Source>
    void CopyIntoNormalized(std::vector<Dest>& out, Dest maximum) const {
        std::transform(
            ViewAs<Source>(), ViewAs<Source>() + NumElements(), std::back_inserter(out), [maximum](const auto& val) {
                return static_cast<Dest>((val / static_cast<double>(std::numeric_limits<Source>::max())) * maximum);
            });
    }

    std::vector<uint8_t> data;
    SizeType width = 0, height = 0, depth = 0;
    ImageElementType elementType = ImageElementType::UINT8;
};

template<uint8_t Dimensions>
struct ImageFrame: AbstractFrame {
    std::vector <ImageChannel<Dimensions>> channels;

    void Read(std::istream& io) override {
        uint8_t numChannels;
        io.read(reinterpret_cast<char*>(&numChannels), sizeof(uint8_t));
        channels.resize(numChannels);
        for (auto c = 0; c < numChannels; ++c) {
            channels[c].Read(io);
        }
    }

    void Write(std::ostream& io) const override {
        const uint8_t numChannels = static_cast<uint8_t>(channels.size());
        io.write(reinterpret_cast<const char*>(&numChannels), sizeof(uint8_t));
        for (auto c = 0; c < numChannels; ++c) {
            channels[c].Write(io);
        }
    }

    [[nodiscard]] std::size_t ByteSize() const override {
        return std::accumulate(channels.begin(), channels.end(), 0,
            [](ImageChannel<Dimensions>& i) -> std::size_t { return i.ByteSize(); });
    }

};

//using Uint8Image2DFrame = ImageFrame<2>;

} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
