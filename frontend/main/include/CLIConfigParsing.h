#pragma once

#include "RuntimeConfig.h"

#include "mmcore/LuaAPI.h"

namespace megamol::frontend {

using megamol::frontend_resources::RuntimeConfig;

RuntimeConfig handle_cli_and_config(const int argc, const char** argv, megamol::core::LuaAPI& lua);

std::vector<std::string> extract_config_file_paths(const int argc, const char** argv);

RuntimeConfig handle_config(RuntimeConfig config, megamol::core::LuaAPI& lua);

RuntimeConfig handle_cli(RuntimeConfig config, const int argc, const char** argv);


}
