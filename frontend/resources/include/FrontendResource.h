/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <any>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <typeinfo>

namespace megamol {
namespace frontend_resources {

template<typename T>
using optional = std::optional<std::reference_wrapper<T>>;

}
namespace frontend {

using megamol::frontend_resources::optional;

class FrontendResource {
public:
    FrontendResource() : identifier{""}, resource{}, type_hash{0}, optional{false} {}

    template<typename T>
    FrontendResource(const char* identifier, const T& resource) : FrontendResource(std::string{identifier}, resource) {}

    template<typename T>
    FrontendResource(const std::string& identifier, const T& resource)
            : identifier{identifier}
            , resource{std::reference_wrapper<const T>(resource)}
            , type_hash{typeid(T).hash_code()}
            , optional{false} {}

    const std::string& getIdentifier() const {
        return identifier;
    }

    template<typename T>
    void setResource(const T& resource) {
        this->resource = std::reference_wrapper<const T>(resource);
    }

    template<typename T>
    T const& getResource() const {
        if (optional) {
            std::cout << "FrontendResource fatal error: resource " + this->identifier +
                             " accessed non-optional but is marked as optional";
            std::cerr << "FrontendResource fatal error: resource " + this->identifier +
                             " accessed non-optional but is marked as optional";
            std::exit(1);
        }

        //try {
        return std::any_cast<std::reference_wrapper<const T>>(resource).get();
        //} catch (const std::bad_any_cast& e) {
        //    return std::nullopt;
        //}
    }

    template<typename T>
    frontend_resources::optional<const T> getOptionalResource() const {
        if (!optional) {
            std::cout << "FrontendResource fatal error: resource " + this->identifier +
                             " accessed optional but is marked as non-optional";
            std::cerr << "FrontendResource fatal error: resource " + this->identifier +
                             " accessed optional but is marked as non-optional";
            std::exit(1);
        }

        if (!resource.has_value()) {
            return std::nullopt;
        }

        //try {
        return std::make_optional(std::any_cast<std::reference_wrapper<const T>>(resource));
        //} catch (const std::bad_any_cast& e) {
        //    return std::nullopt;
        //}
    }

    std::size_t getHash() const {
        return type_hash;
    }

    FrontendResource toOptional() const {
        FrontendResource r = *this;
        r.optional = true;

        return r;
    }

private:
    std::string identifier;
    std::any resource;
    std::size_t type_hash = 0;
    bool optional = false;
};


} /* end namespace frontend */

} /* end namespace megamol */
