/*
 * OpenGL_Context.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>
#include <vector>

namespace megamol {
namespace frontend_resources {

static std::string OpenGL_Context_Req_Name = "OpenGL_Context";

struct OpenGL_Context {
    bool isVersionGEQ(int major, int minor) const;
    bool isExtAvailable(std::string const& ext) const;
    bool areExtAvailable(std::string const& exts) const;
    int major_;
    int minor_;
    std::vector<std::string> ext_;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
