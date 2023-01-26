#ifndef SRC_IMAGESERIES_UTIL_IMAGEUTILS_HPP_
#define SRC_IMAGESERIES_UTIL_IMAGEUTILS_HPP_

#include "vislib/graphics/BitmapImage.h"

#include "glm/glm.hpp"

#include <utility>

namespace megamol::ImageSeries::util {

using Hash = std::uint32_t;

template<typename T>
T combineHash(T a) {
    return a;
}

template<typename T>
T combineHash(T a, T b) {
    return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2));
}

template<typename T, typename... Args>
T combineHash(T a, T b, Args... args) {
    return combineHash(a, combineHash(b, args...));
}

inline Hash hashBytes(const void* data, std::size_t size) {
    Hash hash = 1;
    const char* bytes = static_cast<const char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash += bytes[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    return hash;
}

template<typename T>
Hash computeHash(const T& a) {
    return std::hash<T>()(a);
}

template<>
inline Hash computeHash(const Hash& a) {
    return a;
}

template<typename T>
Hash computeHash(const std::vector<T>& vec) {
    Hash result = 1;
    for (const auto& entry : vec) {
        combineHash(result, computeHash(entry));
    }
    return result;
}

template<typename T, typename... Args>
Hash computeHash(const T& a, const Args&... args) {
    return combineHash(computeHash(a), computeHash(args...));
}

template<typename Type, std::size_t ChannelCount>
class ImageSampler {
public:
    using Vector = typename glm::vec<ChannelCount, float>;

    ImageSampler(const vislib::graphics::BitmapImage& image)
            : pointer(image.PeekDataAs<Type>())
            , width(image.Width())
            , height(image.Height()) {
        if (image.BytesPerPixel() != sizeof(Type) * ChannelCount) {
            throw std::runtime_error("ImageSampler: byte size mismatch");
        }
    }

    Vector get(int x, int y) const {
        Vector result;
        std::size_t index = (clamp(0, x, width - 1) + clamp(0, y, height - 1) * width) * ChannelCount;
        for (std::size_t i = 0; i < ChannelCount; ++i) {
            result[i] = pointer[index + i];
        }
        return result;
    }

    Vector lerp(const glm::vec2& point) const {
        return lerp(point.x, point.y);
    }

    Vector lerp(float x, float y) const {
        float xi, yi;
        float xf = std::modf(x, &xi), yf = std::modf(y, &yi);
        return get(xi, yi) * (1 - xf) * (1 - yf) + get(xi + 1, yi) * xf * (1 - yf) + get(xi, yi + 1) * (1 - xf) * yf +
               get(xi + 1, yi + 1) * xf * yf;
    }

    std::size_t getWidth() const {
        return width;
    }

    std::size_t getHeight() const {
        return height;
    }

private:
    static int clamp(int low, int value, int up) {
        return std::min(std::max(value, low), up);
    }

    const Type* pointer = nullptr;
    std::size_t width = 0;
    std::size_t height = 0;
};

} // namespace megamol::ImageSeries::util

#endif
