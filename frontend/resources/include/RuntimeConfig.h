/*
 * RuntimeConfig.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>
#include <vector>
#include <utility>
#include <optional>

namespace megamol {
namespace frontend_resources {

// make sure that all configuration parameters have sane and useful and EXPLICIT initialization values!
struct RuntimeConfig {
    std::string program_invocation_string = "";

    // general stuff
    using Path = std::string;
    using StringPair = std::pair<std::string/*Config*/, std::string/*Value*/>;

    std::vector<Path> configuration_files = {"megamol_config.lua"};               // set only via (multiple) --config in CLI
    std::vector<std::string> configuration_file_contents = {};
    std::vector<StringPair> cli_options_from_configs = {};                        // mmSetCliOption - set config/option values accepted in CLI
    std::vector<std::string> configuration_file_contents_as_cli = {};
    Path application_directory = "";                                              // mmSetAppDir
    std::vector<Path> resource_directories = {};                                  // mmAddResourceDir
    std::vector<Path> shader_directories = {};                                    // mmAddShaderDir
    Path log_file = "megamol_log.txt";                                            // mmSetLogFile
    unsigned int log_level = 200;                                                 // mmSetLogLevel
    unsigned int echo_level = 200;                                                // mmSetEchoLevel
    std::vector<Path> project_files = {};                                         // NEW: mmLoadProject - project files are loaded after services are up
    std::vector<StringPair> global_values = {}; // use GlobalValueStore resource for access to global values!

    // detailed and service-specific configurations
    // every CLI option can be set via the config file using mmSetConfigValue
    // e.g. "--window 100x200" => mmSetConfigValue("window", "100x200")
    //      "--fullscreen"     => mmSetConfigValue("fullscreen", "on")
    bool interactive = false;
    std::string lua_host_address = "tcp://127.0.0.1:33333";
    bool lua_host_port_retry = true;
    bool opengl_khr_debug = false;
    bool opengl_vsync = false;
    std::optional<std::pair<unsigned int /*width*/,unsigned int /*height*/>> window_size = std::nullopt; // if not set, GLFW service will open window with 3/4 of monitor resolution 
    std::optional<std::pair<unsigned int /*x*/,unsigned int /*y*/>>          window_position = std::nullopt;
    enum WindowMode {
        fullscreen   = 1 << 0,
        nodecoration = 1 << 1,
        topmost      = 1 << 2,
        nocursor     = 1 << 3,
    };
    unsigned int window_mode = 0;
    unsigned int window_monitor = 0;
    bool gui_show = true;
    float gui_scale = 1.0f;
    bool gui_show_entryfbos = false;

    std::string as_string() const {
        auto summarize = [](std::vector<std::string> const& vec) -> std::string {
            std::string result;
            for (auto& s: vec) {
                result += "\n\t\t" + s;
            }
            return result;
        };

        auto print_boolean = [](bool value, std::string const& name) -> std::string {
            // return value ?
        };

        // clang-format off
        return std::string("RuntimeConfig values: "  ) +
            std::string("\n\tProgram invocation: "   ) + "\n\t\t" + program_invocation_string +
            std::string("\n\tConfiguration files: "  ) + summarize(configuration_files) +
            std::string("\n\tApplication directory: ") + application_directory +
            std::string("\n\tResource directories: " ) + summarize(resource_directories) +
            std::string("\n\tShader directories: "   ) + summarize(shader_directories) +
            std::string("\n\tLog file: "             ) + log_file +
            std::string("\n\tLog level: "            ) + std::to_string(log_level) +
            std::string("\n\tEcho level: "           ) + std::to_string(echo_level) +
            std::string("\n\tProject files: "        ) + summarize(project_files) +
            std::string("\n\tLua host address: "     ) + lua_host_address
            ;
            //"\n\t" 
            //lua_host_port_retry = true;
            //opengl_khr_debug = false;
            //opengl_vsync = false;

            //std::optional<std::pair<unsigned int /*width*/,unsigned int /*height*/>> window_size = std::nullopt; // if not set, GLFW service will open window with 3/4 of monitor resolution 
            //std::optional<std::pair<unsigned int /*x*/,unsigned int /*y*/>>          window_position = std::nullopt;
            //enum WindowMode {
            //    fullscreen   = 1 << 0,
            //    nodecoration = 1 << 1,
            //    topmost      = 1 << 2,
            //    nocursor     = 1 << 3,

            //unsigned int window_mode = 0;
            //unsigned int window_monitor = 0;
            //bool interactive = false;
        // clang-format on
    }
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
