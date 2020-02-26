/*
 * Presentations.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "Presentations.h"


using namespace megamol;
using namespace megamol::gui::graph;


// PARAMETER PRESENTATIONS ####################################################

megamol::gui::graph::ParamPresentations::ParamPresentations(void) {}


megamol::gui::graph::ParamPresentations::~ParamPresentations(void) {}


void megamol::gui::graph::ParamPresentations::Present() {

    /*
    auto visitor = [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {

        }
        else if constexpr (std::is_same_v<T, megamol::core::param::ColorParam::ColorType>) {

        }
        else if constexpr (std::is_same_v<T, float>) {

        }
        else if constexpr (std::is_same_v<T, int>) {
            switch (this->type) {
            case (Graph::ParamType::INT): {

            }
            case (Graph::ParamType::ENUM): {

            }
            default:
                break;
            }
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            switch (this->type) {
            case (Graph::ParamType::STRING): {

                break;
            }
            case (Graph::ParamType::TRANSFERFUNCTION): {

                break;
            }
            case (Graph::ParamType::FILEPATH): {

                break;
            }
            case (Graph::ParamType::FLEXENUM): {

                break;
            }
            default:
                break;
            }
        }
        else if constexpr (std::is_same_v<T, vislib::math::Ternary>) {

        }
        else if constexpr (std::is_same_v<T, glm::vec2>) {

        }
        else if constexpr (std::is_same_v<T, glm::vec3>) {

        }
        else if constexpr (std::is_same_v<T, glm::vec4>) {

        }
        else if constexpr (std::is_same_v<T, std::monostate>) {
            switch (this->type) {
            case (Graph::ParamType::BUTTON): {

                break;
            }
            default:
                break;
            }
        }
    };
    std::visit(visitor, this->value);
    */

    /*
    switch (this->type) {
    case (Graph::ParamType::BOOL): {


    } break;
    case (Graph::ParamType::BUTTON): {


    } break;
    case (Graph::ParamType::COLOR): {


    } break;
    case (Graph::ParamType::ENUM): {


    } break;
    case (Graph::ParamType::FILEPATH): {


    } break;
    case (Graph::ParamType::FLEXENUM): {


    } break;
    case (Graph::ParamType::FLOAT): {


    } break;
    case (Graph::ParamType::INT): {


    } break;
    case (Graph::ParamType::STRING): {


    } break;
    case (Graph::ParamType::TERNARY): {


    } break;
    case (Graph::ParamType::TRANSFERFUNCTION): {


    } break;
    case (Graph::ParamType::VECTOR2F): {


    } break;
    case (Graph::ParamType::VECTOR3F): {


    } break;
    case (Graph::ParamType::VECTOR4F): {


    } break;
    default:
        break;
    }
    */
}


// CALL SLOT PRESENTATIONS ####################################################

megamol::gui::graph::CallSlotPresentations::CallSlotPresentations(void) {}


megamol::gui::graph::CallSlotPresentations::~CallSlotPresentations(void) {}


void megamol::gui::graph::CallSlotPresentations::Present() {}


void megamol::gui::graph::CallSlotPresentations::UpdatePosition() {}


// CALL PRESENTATIONS #########################################################

megamol::gui::graph::CallPresentations::CallPresentations(void) {}


megamol::gui::graph::CallPresentations::~CallPresentations(void) {}


void megamol::gui::graph::CallPresentations::Present() {}


// MODULE PRESENTATIONS #######################################################

megamol::gui::graph::ModulePresentations::ModulePresentations(void) {}


megamol::gui::graph::ModulePresentations::~ModulePresentations(void) {}


void megamol::gui::graph::ModulePresentations::Present() {}
