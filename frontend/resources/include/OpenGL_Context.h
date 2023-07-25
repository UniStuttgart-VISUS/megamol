/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <string>
#include <vector>

namespace megamol::frontend_resources {

static std::string OpenGL_Context_Req_Name = "OpenGL_Context";

struct OpenGL_Context {
    bool isVersionGEQ(int major, int minor) const;
    bool isExtAvailable(std::string const& ext) const;
    bool areExtAvailable(std::string const& exts) const;
    int major_;
    int minor_;
    std::vector<std::string> ext_;
};

} // namespace megamol::frontend_resources
