/*
 * MegaMolProject.h
 *
 * Copyright (C) 2022 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace megamol {
namespace frontend_resources {

struct MegaMolProject {
    using path = std::filesystem::path;

    struct ProjectAttributes {
        // example: C:/megamol/project.lua
        std::string project_name = ""; // name of project in some way, either "projcet.lua" or "project" or "something"
        path project_file;             // project.lua
        path project_directory;        // C:/megamol/
    };

    std::optional<ProjectAttributes> attributes = std::nullopt;

    void setProjectFile(path const& file) {
        ProjectAttributes a;
        a.project_file = file;
        a.project_directory = path{file}.remove_filename(); // leaves trailing '/'
        a.project_name = file.filename().stem().string();

        attributes = a;
    }
};
} /* end namespace frontend_resources */
} /* end namespace megamol */
