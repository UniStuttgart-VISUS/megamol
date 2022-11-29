/*
 * AnimationEditor.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "nlohmann/json.hpp"
#include "AnimationData.h"

namespace megamol {
namespace gui {
namespace animation {

void to_json(nlohmann::json& j, const FloatKey& k);
void from_json(const nlohmann::json& j, FloatKey& k);
void to_json(nlohmann::json& j, const StringKey& k);
void from_json(const nlohmann::json& j, StringKey& k);
void to_json(nlohmann::json& j, const Vec3Key& k);
void from_json(const nlohmann::json& j, Vec3Key& k);

void to_json(nlohmann::json& j, const FloatAnimation& f);
void from_json(const nlohmann::json& j, FloatAnimation& f);
void to_json(nlohmann::json& j, const StringAnimation& s);
void from_json(const nlohmann::json& j, StringAnimation& s);
void to_json(nlohmann::json& j, const Vec3Animation& v);
void from_json(const nlohmann::json& j, Vec3Animation& s);


// PD code from https://github.com/ocornut/imgui/issues/1496#issuecomment-1287772456
void BeginGroupPanel(const char* name, const ImVec2& size);
void EndGroupPanel();

std::vector<float> GetFloats(std::string vector_string);

} // namespace animation
} // namespace gui
} // namespace megamol
