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
void to_json(nlohmann::json& j, const FloatAnimation& f);
void from_json(const nlohmann::json& j, FloatAnimation& f);

} // namespace animation
} // namespace gui
} // namespace megamol
