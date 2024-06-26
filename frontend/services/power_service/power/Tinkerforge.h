#pragma once

#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace megamol::power {
/// <summary>
/// Parses a file into a JSON object.
/// </summary>
/// <param name="path">Path to the JSON file.</param>
/// <returns>JSON object of the file.</returns>
nlohmann::json parse_json_file(std::filesystem::path const& path) {
    std::ifstream f(path);
    auto json_file = nlohmann::json::parse(f);
    f.close();
    return json_file;
}

/// <summary>
/// Look-up of tinkerforge sensor name in the given JSON object.
/// </summary>
/// <param name="data">JSON object with tinkerforge sensor names.</param>
/// <param name="search">Tinkerforge UID as search string.</param>
/// <returns>Name of the sensor.</returns>
/// <exception cref="std::runtime_error"></exception>
std::string get_tf_name(nlohmann::json const& data, std::string const& search) {
    for (auto const& el : data.items()) {
        // expecting array
        if (!el.value().is_array()) {
            throw std::runtime_error("expected array");
        }

        int counter = 0;
        for (auto const& val : el.value().items()) {
            auto const v = val.value().get<std::string>();
            if (v == search) {
                if (el.value().size() > 1) {
                    return el.key() + std::to_string(counter);
                } else {
                    return el.key();
                }
            }
            ++counter;
        }
    }
    return "";
}

/// <summary>
/// Extracts UID from power-overwhelming tinkerforge name.
/// </summary>
/// <param name="name">Tinkerforge name.</param>
/// <returns>UID of the tinkerforge sensor.</returns>
std::string get_search_string(std::string const& name) {
    auto const reg = std::regex(R"(^Tinkerforge/[\w|\:]+/(\w+)$)");
    std::smatch match;
    if (std::regex_match(name, match, reg)) {
        return match[1];
    }
    return "";
}

/// <summary>
/// Get name of the tinkerforge sensor from the JSON object.
/// </summary>
/// <param name="data">JSON object.</param>
/// <param name="name">Power-overwhelming name of the tinkerforge sensor.</param>
/// <returns>Name of tinkerforge sensor.</returns>
/// <exception cref="std::runtime_error"></exception>
std::string transform_tf_name(nlohmann::json const& data, std::string const& name) {
    auto const transformed_name = get_tf_name(data, get_search_string(name));
    if (transformed_name.empty()) {
        throw std::runtime_error("unable to transform name");
    }
    return transformed_name;
}

} // namespace megamol::power
