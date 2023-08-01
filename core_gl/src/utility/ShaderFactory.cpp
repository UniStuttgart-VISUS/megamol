/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore_gl/utility/ShaderFactory.h"

#include <msf/LineTranslator.h>
#include <msf/ShaderFactory.h>
#include <msf/ShaderFactoryUtils.h>

static msf::ShaderFactory msf_factory;

msf::ShaderFactoryOptionsOpenGL megamol::core::utility::make_path_shader_options(
    megamol::frontend_resources::RuntimeConfig const& conf) {
    std::vector<std::filesystem::path> paths;
    paths.reserve(conf.shader_directories.size());
    for (auto const& path : conf.shader_directories) {
        paths.emplace_back(path);
    }
    return msf::ShaderFactoryOptionsOpenGL(paths);
}

glowl::GLSLProgram::ShaderSourceList::value_type megamol::core::utility::make_glowl_shader_source(
    std::filesystem::path const& shader_source_path, msf::ShaderFactoryOptionsOpenGL const& options,
    msf::LineTranslator& translator) {
    auto const shader_string = translator.cleanupShader(msf_factory.preprocess(shader_source_path, options));
    auto const type = msf::getShaderTypeInt(shader_source_path);

    return std::make_pair(static_cast<glowl::GLSLProgram::ShaderType>(type), shader_string);
}
