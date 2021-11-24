#pragma once
#include <functional>

namespace megamol {
namespace frontend_resources {

    struct common_types {
        using lua_func_type = std::function<std::tuple<bool, std::string>(std::string const&)>;
    };

}
}
