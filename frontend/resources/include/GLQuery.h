#pragma once

#include <cstdint>

namespace megamol::frontend_resources::performance {
class GLQuery {
public:
    GLQuery();

    ~GLQuery();

    void Counter() const;

    uint64_t GetNW();

private:
    uint32_t handle_;
    uint64_t value_ = 0;
};
} // namespace megamol::frontend_resources::performance
