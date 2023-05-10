/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <nlohmann/json.hpp>

#include "mmcore/utility/animation/AnimationData.h"

namespace megamol::core::utility::animation {

void to_json(nlohmann::json& j, const FloatKey& k);
void from_json(const nlohmann::json& j, FloatKey& k);
void to_json(nlohmann::json& j, const StringKey& k);
void from_json(const nlohmann::json& j, StringKey& k);
void to_json(nlohmann::json& j, const VectorKey<FloatKey>& k);
void from_json(const nlohmann::json& j, VectorKey<FloatKey>& k);

void to_json(nlohmann::json& j, const FloatAnimation& f);
void from_json(const nlohmann::json& j, FloatAnimation& f);
void to_json(nlohmann::json& j, const StringAnimation& s);
void from_json(const nlohmann::json& j, StringAnimation& s);
void to_json(nlohmann::json& j, const FloatVectorAnimation& v);
void from_json(const nlohmann::json& j, FloatVectorAnimation& s);


// this does not seem okay
// PD code from https://github.com/ocornut/imgui/issues/1496#issuecomment-1287772456
#if 0
void BeginGroupPanel(const char* name, const ImVec2& size);
void EndGroupPanel();
#endif

std::vector<float> GetFloats(std::string vector_string);

} // namespace megamol::core::utility::animation
