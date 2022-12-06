/**
 * MegaMol
 * Copyright (c) 2012, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/ResourceWrapper.h"

#include <fstream>
#include <sstream>

using namespace megamol::core;
using namespace megamol::core::utility;

using megamol::core::utility::log::Log;

std::filesystem::path ResourceWrapper::GetResourcePath(
    frontend_resources::RuntimeConfig const& runtimeConf, const std::string& filename) {
    for (const auto& resource_directory : runtimeConf.resource_directories) {
        auto path = std::filesystem::path(resource_directory) / std::filesystem::path(filename);
        if (std::filesystem::is_regular_file(path)) {
            return path;
        }
    }
    throw std::runtime_error("Resource file not found: " + filename);
}

std::vector<char> ResourceWrapper::LoadResource(
    frontend_resources::RuntimeConfig const& runtimeConf, const std::string& filename) {
    const auto path = GetResourcePath(runtimeConf, filename);

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot read resource file \"" + path.string() + "\"!");
    }
    file.seekg(0, std::ios::end);
    auto filesize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buf(filesize);
    file.read(buf.data(), filesize);
    return buf;
}

std::string ResourceWrapper::LoadTextResource(
    frontend_resources::RuntimeConfig const& runtimeConf, const std::string& filename) {
    const auto path = GetResourcePath(runtimeConf, filename);

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot read resource file \"" + path.string() + "\"!");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
