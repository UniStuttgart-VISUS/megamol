/*
 * LuaScriptPaths.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace megamol {
namespace frontend_resources {

static std::string LuaScriptPaths_Req_Name = "LuaScriptPaths";
static std::string SetLuaScriptPath_Req_Name = "SetLuaScriptPath";

using set_lua_script_path_func = std::function<void(std::string const&)>;

struct LuaScriptPaths {
    std::vector<std::string> lua_script_paths;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
