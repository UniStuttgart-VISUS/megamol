namespace megamol {
namespace datatools {
namespace io {
namespace dataformat {

class ImageElementType {
public:
    enum Value : uint8_t {
        UINT8 = 1,
        UINT16 = 2,
        UINT32 = 3, // This is a semantic alias! OK?
        RGBA8 = 3,  // This is a semantic alias! OK?
        FLOAT = 4,
        DOUBLE = 5
    };
    ImageElementType() = default;
    constexpr ImageElementType(Value ct) : value(ct) {}
    constexpr operator Value() const {
        return value;
    }
    explicit operator bool() = delete;
    void Set(uint8_t other) {
        if (other < 1 || other > 5)
            throw std::invalid_argument("value not supported");
        value = static_cast<Value>(other);
    }
    //constexpr operator uint8_t() const {
    //    return static_cast<uint8_t>(value);
    //}

    [[nodiscard]] constexpr std::size_t ByteSize() const {
        switch (value) {
        case UINT8:
            return sizeof(uint8_t);
        case UINT16:
            return sizeof(uint16_t);
        case UINT32:
            // AKA case RGBA8:
            return sizeof(uint32_t);
        case FLOAT:
            return sizeof(float);
        case DOUBLE:
            return sizeof(double);
        }
        return 0;
    }

private:
    Value value = UINT8;
};


} // namespace dataformat
} // namespace io
} // namespace datatools
} // namespace megamol
