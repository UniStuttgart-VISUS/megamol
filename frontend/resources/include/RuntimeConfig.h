/*
 * RuntimeConfig.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mmcore/utility/buildinfo/BuildInfo.h"

namespace megamol {
namespace frontend_resources {

// make sure that all configuration parameters have sane and useful and EXPLICIT initialization values!
struct RuntimeConfig {
    std::string program_invocation_string = "";

    // general stuff
    using Path = std::string;
    using StringPair = std::pair<std::string /*Config*/, std::string /*Value*/>;
    using UintPair = std::pair<unsigned int, unsigned int>;

    std::vector<Path> configuration_files = {"megamol_config.lua"}; // set only via (multiple) --config in CLI
    std::vector<std::string> configuration_file_contents = {};
    std::vector<StringPair> cli_options_from_configs = {}; // mmSetCliOption - set config/option values accepted in CLI
    std::vector<std::string> configuration_file_contents_as_cli = {};
    Path megamol_executable_directory = "";      // mmGetMegaMolExecutableDirectory
    Path application_directory = "";             // mmSetAppDir
    std::vector<Path> resource_directories = {}; // mmAddResourceDir
    std::vector<Path> shader_directories = {};   // mmAddShaderDir
    Path log_file = "megamol_log.txt";           // mmSetLogFile
    unsigned int log_level = 200;                // mmSetLogLevel
    unsigned int echo_level = 200;               // mmSetEchoLevel
    std::vector<Path> project_files = {};        // NEW: mmLoadProject - project files are loaded after services are up
    std::vector<StringPair> global_values = {};  // use GlobalValueStore resource for access to global values!
    std::string cli_execute_lua_commands;

    // detailed and service-specific configurations
    // every CLI option can be set via the config file using mmSetConfigValue
    // e.g. "--window 100x200" => mmSetCliOption("window", "100x200")
    //      "--fullscreen"     => mmSetCliOption("fullscreen", "on")
    bool interactive = false;
    std::string lua_host_address = "tcp://127.0.0.1:33333";
    bool lua_host_port_retry = true;
    // Different default values for a present openGL
    // In the WITH_GL not defined case, the user cannot overwrite the default
#ifdef WITH_GL
    bool no_opengl = false;
#else
    bool no_opengl = true;
#endif
    bool opengl_khr_debug = false;
    bool opengl_vsync = false;
    std::optional<std::tuple<unsigned int /*major*/, unsigned int /*minor*/, bool /*true=>core, false=>compat*/>>
        opengl_context_version = {{4, 6, false /*compat*/}};
    std::optional<UintPair> window_size =
        std::nullopt; // if not set, GLFW service will open window with 3/4 of monitor resolution
    std::optional<UintPair> window_position = std::nullopt;
    enum WindowMode {
        fullscreen = 1 << 0,
        nodecoration = 1 << 1,
        topmost = 1 << 2,
        nocursor = 1 << 3,
    };
    unsigned int window_mode = 0;
    unsigned int window_monitor = 0;
    bool force_window_size = false;
    bool gui_show = true;
    float gui_scale = 1.0f;
    bool screenshot_show_privacy_note = true;
    bool show_version_note = true;
    std::string profiling_output_file;

    struct Tile {
        UintPair global_framebuffer_resolution; // e.g. whole powerwall resolution, needed for tiling
        UintPair tile_start_pixel;
        UintPair tile_resolution;
    };
    std::optional<Tile> local_viewport_tile = std::nullopt; // defaults to local framebuffer == local tile

    // e.g. window resolution or powerwall projector resolution, will be applied to all views/entry points
    std::optional<UintPair> local_framebuffer_resolution = std::nullopt;

    bool remote_headnode = false;
    bool remote_rendernode = false;
    bool remote_mpirendernode = false;
    bool remote_headnode_broadcast_quit = false;
    bool remote_headnode_broadcast_initial_project = false;
    bool remote_headnode_connect_on_start = false;
    unsigned int remote_mpi_broadcast_rank = 0;
    std::string remote_headnode_zmq_target_address = "tcp://127.0.0.1:62562";
    std::string remote_rendernode_zmq_source_address = "tcp://*:62562";

    enum class VRMode {
        Off,
#ifdef WITH_VR_SERVICE_UNITY_KOLABBW
        UnityKolab,
#endif // WITH_VR_SERVICE_UNITY_KOLABBW
    };
    VRMode vr_mode = VRMode::Off;

    std::string as_string() const {
        auto summarize = [](std::vector<std::string> const& vec) -> std::string {
            std::string result;
            for (auto& s : vec) {
                result += "\n\t\t" + s;
            }
            return result;
        };

        auto print_boolean = [](bool value, std::string const& name) -> std::string {
            // return value ?
        };

        // clang-format off
        return std::string("RuntimeConfig values: "  ) +
            std::string("\n\tExecutable directory: "   ) + "\n\t\t" + megamol_executable_directory +
            std::string("\n\tProgram invocation: "   ) + "\n\t\t" + program_invocation_string +
            std::string("\n\tVersion: "              ) + megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH() +
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
