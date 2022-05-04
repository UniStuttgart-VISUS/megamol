/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#include "mmcore/utility/buildinfo/BuildInfo.h"

#include <string>

#include "cmrc/cmrc.hpp"

CMRC_DECLARE(megamol_build_info_rc);

namespace {
cmrc::file loadResource(std::string const& resourceName) {
    static auto resources = cmrc::megamol_build_info_rc::get_filesystem();
    if (!resources.is_file(resourceName)) {
        throw std::runtime_error("Missing resource: " + resourceName);
    }
    return resources.open(resourceName);
}

std::string loadStringResource(std::string const& resourceName) {
    auto res = loadResource(resourceName);
    return std::string(res.begin(), res.end());
}

int loadIntResource(std::string const& resourceName) {
    auto res = loadStringResource(resourceName);
    return std::stoi(res);
}

uint64_t loadUInt64Resource(std::string const& resourceName) {
    auto res = loadStringResource(resourceName);
    return std::stoull(res);
}
} // namespace

int megamol::core::utility::buildinfo::MEGAMOL_VERSION_MAJOR() {
    static int res = loadIntResource("MEGAMOL_VERSION_MAJOR");
    return res;
}

int megamol::core::utility::buildinfo::MEGAMOL_VERSION_MINOR() {
    static int res = loadIntResource("MEGAMOL_VERSION_MINOR");
    return res;
}

int megamol::core::utility::buildinfo::MEGAMOL_VERSION_PATCH() {
    static int res = loadIntResource("MEGAMOL_VERSION_PATCH");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_VERSION() {
    static auto res = loadStringResource("MEGAMOL_VERSION");
    return res;
}

uint64_t megamol::core::utility::buildinfo::MEGAMOL_BUILD_TIMESTAMP() {
    static uint64_t res = loadUInt64Resource("MEGAMOL_BUILD_TIMESTAMP");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_BUILD_TIME() {
    static auto res = loadStringResource("MEGAMOL_BUILD_TIME");
    return res;
}

bool megamol::core::utility::buildinfo::MEGAMOL_BUILD_TIME_IS_EXACT() {
    static bool res = static_cast<bool>(loadIntResource("MEGAMOL_BUILD_TIME_IS_EXACT"));
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH() {
    static auto res = loadStringResource("MEGAMOL_GIT_HASH");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_BRANCH_NAME() {
    static auto res = loadStringResource("MEGAMOL_GIT_BRANCH_NAME");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_BRANCH_NAME_FULL() {
    static auto res = loadStringResource("MEGAMOL_GIT_BRANCH_NAME_FULL");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_REMOTE_NAME() {
    static auto res = loadStringResource("MEGAMOL_GIT_REMOTE_NAME");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_REMOTE_URL() {
    static auto res = loadStringResource("MEGAMOL_GIT_REMOTE_URL");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_DIFF() {
    static auto res = loadStringResource("MEGAMOL_GIT_DIFF");
    return res;
}

int megamol::core::utility::buildinfo::MEGAMOL_GIT_IS_DIRTY() {
    static int res = loadIntResource("MEGAMOL_GIT_IS_DIRTY");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_LICENSE() {
    static auto res = loadStringResource("MEGAMOL_LICENSE");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_CMAKE_CACHE() {
    static auto res = loadStringResource("MEGAMOL_CMAKE_CACHE");
    return res;
}
