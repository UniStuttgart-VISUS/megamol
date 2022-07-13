/*
 * ShaderFactory.cpp
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#include "mmcore_gl/utility/ShaderFactory.h"

#include "msf/LineTranslator.h"
#include "msf/ShaderFactory.h"
#include "msf/ShaderFactoryUtils.h"

static msf::ShaderFactory msf_factory;

glowl::GLSLProgram::ShaderSourceList::value_type megamol::core::utility::make_glowl_shader_source(
    std::filesystem::path const& shader_source_path, msf::ShaderFactoryOptionsOpenGL const& options,
    msf::LineTranslator& translator) {
    auto const shader_string = translator.cleanupShader(msf_factory.preprocess(shader_source_path, options));
    auto const type = msf::getShaderTypeInt(shader_source_path);

    return std::make_pair(static_cast<glowl::GLSLProgram::ShaderType>(type), shader_string);
}
