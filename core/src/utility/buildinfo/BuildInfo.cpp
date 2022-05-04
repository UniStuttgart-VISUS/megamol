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
    static const auto resources = cmrc::megamol_build_info_rc::get_filesystem();
    if (!resources.is_file(resourceName)) {
        throw std::runtime_error("Missing resource: " + resourceName);
    }
    return resources.open(resourceName);
}

std::string loadStringResource(std::string const& resourceName) {
    const auto res = loadResource(resourceName);
    return std::string(res.begin(), res.end());
}

int loadIntResource(std::string const& resourceName) {
    const auto res = loadStringResource(resourceName);
    return std::stoi(res);
}

uint64_t loadUInt64Resource(std::string const& resourceName) {
    const auto res = loadStringResource(resourceName);
    return std::stoull(res);
}
} // namespace

int megamol::core::utility::buildinfo::MEGAMOL_VERSION_MAJOR() {
    static const int res = loadIntResource("MEGAMOL_VERSION_MAJOR");
    return res;
}

int megamol::core::utility::buildinfo::MEGAMOL_VERSION_MINOR() {
    static const int res = loadIntResource("MEGAMOL_VERSION_MINOR");
    return res;
}

int megamol::core::utility::buildinfo::MEGAMOL_VERSION_PATCH() {
    static const int res = loadIntResource("MEGAMOL_VERSION_PATCH");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_VERSION() {
    static const auto res = loadStringResource("MEGAMOL_VERSION");
    return res;
}

uint64_t megamol::core::utility::buildinfo::MEGAMOL_BUILD_TIMESTAMP() {
    static const uint64_t res = loadUInt64Resource("MEGAMOL_BUILD_TIMESTAMP");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_BUILD_TIME() {
    static const auto res = loadStringResource("MEGAMOL_BUILD_TIME");
    return res;
}

bool megamol::core::utility::buildinfo::MEGAMOL_BUILD_TIME_IS_EXACT() {
    static const bool res = static_cast<bool>(loadIntResource("MEGAMOL_BUILD_TIME_IS_EXACT"));
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH() {
    static const auto res = loadStringResource("MEGAMOL_GIT_HASH");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_BRANCH_NAME() {
    static const auto res = loadStringResource("MEGAMOL_GIT_BRANCH_NAME");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_BRANCH_NAME_FULL() {
    static const auto res = loadStringResource("MEGAMOL_GIT_BRANCH_NAME_FULL");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_REMOTE_NAME() {
    static const auto res = loadStringResource("MEGAMOL_GIT_REMOTE_NAME");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_REMOTE_URL() {
    static const auto res = loadStringResource("MEGAMOL_GIT_REMOTE_URL");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_GIT_DIFF() {
    static const auto res = loadStringResource("MEGAMOL_GIT_DIFF");
    return res;
}

int megamol::core::utility::buildinfo::MEGAMOL_GIT_IS_DIRTY() {
    static const int res = loadIntResource("MEGAMOL_GIT_IS_DIRTY");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_LICENSE() {
    static const auto res = loadStringResource("MEGAMOL_LICENSE");
    return res;
}

std::string const& megamol::core::utility::buildinfo::MEGAMOL_CMAKE_CACHE() {
    static const auto res = loadStringResource("MEGAMOL_CMAKE_CACHE");
    return res;
}
