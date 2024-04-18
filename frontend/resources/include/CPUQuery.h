#pragma once

#include <AnyQuery.h>

namespace megamol::frontend_resources::performance {
/// <summary>
/// Wrapper for CPU timer query.
/// </summary>
class CPUQuery : public AnyQuery {
public:
    CPUQuery();

    ~CPUQuery() override;

    /// <summary>
    /// Set timestamp query.
    /// </summary>
    void Counter() override;

    std::shared_ptr<AnyQuery> MakeAnother() override {
        return std::make_shared<CPUQuery>();
    }

    /// <summary>
    /// Try to retrieve the timestamp.
    /// Does not wait if value is not ready.
    /// After successful retrieval will return acquired timestamp and not try again.
    /// </summary>
    /// <returns>Queried timestamp or zero if value is not ready</returns>
    time_point GetNW() override;
};
} // namespace megamol::frontend_resources::performance
