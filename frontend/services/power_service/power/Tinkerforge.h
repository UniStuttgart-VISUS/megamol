#pragma once

#include <filesystem>
#include <fstream>
#include <regex>

#include <nlohmann/json.hpp>

namespace megamol::power {
nlohmann::json parse_json_file(std::filesystem::path const& path) {
    try {
        std::ifstream f(path);
        auto json_file = nlohmann::json::parse(f);
        f.close();
        return json_file;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Could not open JSON file {}", path.string());
        throw;
    }
}

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

std::string get_search_string(std::string const& name) {
    auto const reg = std::regex(R"(^Tinkerforge/[\w|\:]+/(\w+)$)");
    std::smatch match;
    if (std::regex_match(name, match, reg)) {
        return match[1];
    }
    return "";
}

std::string transform_tf_name(nlohmann::json const& data, std::string const& name) {
    auto const transformed_name = get_tf_name(data, get_search_string(name));
    if (transformed_name.empty()) {
        throw std::runtime_error("unable to transform name");
    }
    return transformed_name;
}

} // namespace megamol::power
