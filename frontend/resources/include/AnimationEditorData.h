/*
 * AnimationEditorData.h
 *
 * Copyright (C) 2022 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/utility/animation/AnimationData.h"

namespace megamol::frontend_resources {

static std::string AnimationEditorData_Req_Name = "AnimationEditorData";

struct AnimationEditorData {
    core::utility::animation::FloatVectorAnimation* pos_animation = nullptr;
    core::utility::animation::FloatVectorAnimation* orientation_animation = nullptr;
};

} // namespace megamol::frontend_resources
