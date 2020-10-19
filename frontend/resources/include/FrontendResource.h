/*
 * FrontendResource.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <string>
#include <any>
#include <functional>
#include <optional>

namespace megamol {
namespace frontend {


class FrontendResource {
public:
    FrontendResource()
        : identifier{""}
        , resource{}
    {}

    template <typename T>
    FrontendResource(const char* identifier, const T& resource) : FrontendResource(std::string{identifier}, resource) {}

    template <typename T>
    FrontendResource(const std::string& identifier, const T& resource)
        : identifier{identifier}, resource{std::reference_wrapper<const T>(resource)} {}

    const std::string& getIdentifier() const { return identifier; }

    template <typename T> void setResource(const T& resource) {
        this->resource = std::reference_wrapper<const T>(resource);
    }

    template <typename T> T const& getResource() const {
        //try {
            return std::any_cast<std::reference_wrapper<const T>>(resource).get();
        //} catch (const std::bad_any_cast& e) {
        //    return std::nullopt;
        //}
    }

private:
    std::string identifier;
    std::any resource;
};


} /* end namespace frontend */

} /* end namespace megamol */

