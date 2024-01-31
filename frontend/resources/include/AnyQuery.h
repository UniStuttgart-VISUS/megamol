#pragma once

#include <TimeTypes.h>

namespace megamol::frontend_resources::performance {
/// <summary>
/// Wrapper for generic timer query.
/// </summary>
class AnyQuery {
public:
    AnyQuery() = default;

    virtual ~AnyQuery() = default;

    /// <summary>
    /// Set timestamp query.
    /// </summary>
    virtual void Counter() = 0;

    /// <summary>
    /// Try to retrieve the timestamp.
    /// Does not wait if value is not ready.
    /// After successful retrieval will return acquired timestamp and not try again.
    /// </summary>
    /// <returns>Queried timestamp or zero if value is not ready</returns>
    virtual time_point GetNW() = 0;

protected:
    time_point value_ = zero_time;
};
} // namespace megamol::frontend_resources::performance
