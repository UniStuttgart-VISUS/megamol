/*
 * RuntimeConfig.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>
#include <vector>
#include <map>

namespace megamol {
namespace frontend_resources {

// make sure that all configuration parameters have sane and useful and EXPLICIT initialization values!
struct RuntimeConfig {
    std::string program_invocation_string = "";

    // general stuff
    using Path = std::string;
    Path m_configuration_file = "";
    Path m_application_directory = "";
    std::vector<Path> m_resource_directories = {};
    std::vector<Path> m_shader_directories = {};
    std::map<std::string/*Key*/, std::string/*Value*/> m_config_values = {};
    std::vector<Path> project_files = {};

    // service-specific configurations
    std::string lua_host_address = "tcp://127.0.0.1:33333";
    bool lua_host_port_retry = true;
    bool opengl_khr_debug = false;
    bool opengl_vsync = false;
    std::vector<unsigned int> window_size = {}; // if not set, GLFW service will open window with 3/4 of monitor resolution 
    std::vector<unsigned int> window_position = {};
    enum WindowMode {
        fullscreen   = 1 << 0,
        nodecoration = 1 << 1,
        topmost      = 1 << 2,
        nocursor     = 1 << 3,
    };
    unsigned int window_mode = 0;
    unsigned int window_monitor = 0;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
