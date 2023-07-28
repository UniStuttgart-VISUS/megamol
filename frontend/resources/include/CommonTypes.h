/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <functional>
#include <string>

namespace megamol::frontend_resources {

struct common_types {
    using lua_func_type = std::function<std::tuple<bool, std::string>(std::string const&)>;
};

} // namespace megamol::frontend_resources
