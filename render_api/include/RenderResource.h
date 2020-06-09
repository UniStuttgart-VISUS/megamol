/*
 * RenderResource.h
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
namespace render_api {


class RenderResource {
template <typename T> using OptionalConstReference = std::optional<std::reference_wrapper<const T>>;
public:
    RenderResource()
		: identifier{""}
		, resource{}
	{}

    template <typename T>
    RenderResource(const char* identifier, const T& resource) : RenderResource(std::string{identifier}, resource) {}

    template <typename T>
    RenderResource(const std::string& identifier, const T& resource)
        : identifier{identifier}, resource{std::reference_wrapper<const T>(resource)} {}

    const std::string& getIdentifier() const { return identifier; }

    template <typename T> void setResource(const T& resource) {
        this->resource = std::reference_wrapper<const T>(resource);
    }

    template <typename T> OptionalConstReference<T> getResource() const {
        try {
            return std::make_optional( std::any_cast<std::reference_wrapper<const T>>(resource) );
        } catch (const std::bad_any_cast& e) {
            return std::nullopt;
        }
    }

private:
    std::string identifier;
    std::any resource;
};


} /* end namespace render_api */

} /* end namespace megamol */

