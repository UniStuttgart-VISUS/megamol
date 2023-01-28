/*
 * JSONHelper.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_UTILITY_JSONHELPER_INCLUDED
#define MEGAMOL_UTILITY_JSONHELPER_INCLUDED

#include <string>

#include <nlohmann/json.hpp>

#include "mmcore/utility/log/Log.h"

namespace megamol::core::utility {

/**
 * Read value from given JSON node
 */
template<typename T>
bool get_json_value(
    const nlohmann::json& in_json, const std::vector<std::string>& in_nodes, T* out_value, size_t array_size = 0) {

    try {
        std::string node_name;
        auto json_value = in_json;
        auto node_count = in_nodes.size();
        if (node_count != 0) {
            node_name = in_nodes.front();
            json_value = in_json.at(in_nodes.front());
            for (size_t i = 1; i < node_count; i++) {
                node_name = node_name + "/" + in_nodes[i];
                json_value = json_value.at(in_nodes[i]);
            }
        }
        if (array_size > 0) {
            if (!json_value.is_array()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "JSON ERROR - %s is no JSON array. [%s, %s, line %d]\n", node_name.c_str(), __FILE__, __FUNCTION__,
                    __LINE__);
                return false;
            }
            if (json_value.size() != array_size) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "JSON ERROR - %s is no JSON array of size %i. [%s, %s, line %d]\n", node_name.c_str(), array_size,
                    __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            for (size_t i = 0; i < array_size; i++) {
                if constexpr (std::is_same_v<T, bool>) {
                    if (json_value[i].is_boolean()) {
                        out_value[i] = json_value[i];
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "JSON ERROR - Couldn't read 'bool' value from json node '%s'. [%s, %s, line %d]\n",
                            node_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        return false;
                    }
                } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                    if (json_value[i].is_number()) {
                        out_value[i] = json_value[i];
                    } else {
                        bool nan_case = false;
                        if (json_value[i].is_string()) {
                            if (json_value[i] == "null") {
                                out_value[i] = std::numeric_limits<T>::quiet_NaN();
                                nan_case = true;
                            }
                        }
                        if (!nan_case) {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "JSON ERROR - Couldn't read 'float' or 'int' value from json node '%s'. [%s, %s, line "
                                "%d]\n",
                                node_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                            return false;
                        }
                    }
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "JSON ERROR - Unknown value type for array. [%s, %s, line %d]\n", node_name.c_str(), __FILE__,
                        __FUNCTION__, __LINE__);
                    return false;
                }
            }
        } else {
            if constexpr (std::is_same_v<T, bool>) {
                if (json_value.is_boolean()) {
                    json_value.get_to((*out_value));
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "JSON ERROR - Couldn't read 'bool' value from json node '%s'. [%s, %s, line %d]\n",
                        node_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
            } else if constexpr (std::is_same_v<T, std::string>) {
                if (json_value.is_string()) {
                    std::string value;
                    json_value.get_to((*out_value));
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "JSON ERROR - Couldn't read 'string' value from json node '%s'. [%s, %s, line %d]\n",
                        node_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                    return false;
                }
            } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int> ||
                                 std::is_same_v<T, unsigned int>) {
                if (json_value.is_number()) {
                    json_value.get_to((*out_value));
                } else {
                    bool nan_case = false;
                    if (json_value.is_string()) {
                        if (json_value == "null") {
                            (*out_value) = std::numeric_limits<T>::quiet_NaN();
                            nan_case = true;
                        }
                    }
                    if (!nan_case) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "JSON ERROR - Couldn't read 'float' or 'int' value from json node '%s'. [%s, %s, line "
                            "%d]\n",
                            node_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
                        return false;
                    }
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "JSON ERROR - Unknown value type. [%s, %s, line %d]\n", node_name.c_str(), __FILE__, __FUNCTION__,
                    __LINE__);
                return false;
            }
        }
        return true;
    } catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON TYPE ERROR: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (nlohmann::json::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "JSON: %s. [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "JSON ERROR - Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
}

} // namespace megamol::core::utility

#endif // MEGAMOL_UTILITY_JSONHELPER_INCLUDED
