/*
 * GlobalValueStore.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <map>
#include <optional>
#include <string>

namespace megamol {
namespace frontend_resources {

struct GlobalValueStore {
    std::map<std::string /*Key*/, std::string /*Value*/> key_values = {}; // mmSetGlobalValue + mmGetGlobalValue

    // add or update a key-value pair
    void insert(std::string const& key, std::string const& value) {
        key_values.insert_or_assign(key, value);
    }

    // retrieve value for given key. if key is present, the optional holds the value.
    std::optional<std::string> maybe_get(std::string const& key) const {
        auto value_it = key_values.find(key);
        if (value_it != key_values.end()) {
            return std::optional{value_it->second};
        } else {
            return std::nullopt;
        }
    }

    std::string as_string() const {
        auto summarize_globals = [&]() -> std::string {
            std::string result;
            for (auto& kv : key_values) {
                result += "\n\t\t" + kv.first + " : " + kv.second;
            }
            return result;
        };

        return std::string("\n\tGlobal Key-Values: ") + summarize_globals();
    }
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
