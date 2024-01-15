#pragma once

#include <cstdint>

namespace megamol::frontend_resources::performance {
/// <summary>
/// Wrapper for OpenGL timer query.
/// </summary>
class GLQuery {
public:
    GLQuery();

    ~GLQuery();

    /// <summary>
    /// Set timestamp query.
    /// </summary>
    void Counter() const;

    /// <summary>
    /// Try to retrieve the timestamp.
    /// Does not wait if value is not ready.
    /// After successfull retrieval will return acquired timestamp and not try again.
    /// </summary>
    /// <returns>Queried timestamp or zero if value is not ready</returns>
    uint64_t GetNW();

private:
    uint32_t handle_;
    uint64_t value_ = 0;
};
} // namespace megamol::frontend_resources::performance
