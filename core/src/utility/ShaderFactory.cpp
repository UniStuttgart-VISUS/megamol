/*
 * ShaderFactory.cpp
 *
 * Copyright (C) 2020-2021 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/utility/ShaderFactory.h"

#include "msf/LineTranslator.h"
#include "msf/ShaderFactory.h"
#include "msf/ShaderFactoryUtils.h"

static msf::ShaderFactory msf_factory;

std::unique_ptr<glowl::GLSLProgram> megamol::core::utility::make_glowl_shader(std::string const& label,
    msf::ShaderFactoryOptionsOpenGL const& options, const std::vector<std::filesystem::path>& paths) {

    glowl::GLSLProgram::ShaderSourceList source_list;
    std::vector<msf::LineTranslator> translators;

    // TODO: BUG!!! All translators will use the same ids for different files!!!
    for (const auto& path : paths) {
        auto const shader_string = translators.emplace_back(msf_factory.preprocess(path, options)).getCleanShader();
        auto const type = msf::getShaderTypeInt(path);
        source_list.emplace_back(std::make_pair(static_cast<glowl::GLSLProgram::ShaderType>(type), shader_string));
    }

    try {
        auto program = std::make_unique<glowl::GLSLProgram>(source_list);
        program->setDebugLabel(label);
        return program;
    } catch (const glowl::GLSLProgramException& e) {
        std::string error = e.what();
        for (const auto& t : translators) {
            error = t.translateErrorLog(error);
        }
        throw glowl::GLSLProgramException(error);
    }
}
