#include "mmcore/view/Call6DofInteraction.h"


/*
 * megamol::pcl::CallPcd::FunctionName
 */
const char* megamol::core::view::Call6DofInteraction::FunctionName(unsigned int idx) {
    if (idx < Call6DofInteraction::FunctionCount()) {
        return Call6DofInteraction::INTENTS[idx].data();
    } else {
        return "";
    }
}


/*
 * megamol::core::misc::VolumetricDataCall::IDX_GET_STATE
 */
const unsigned int megamol::core::view::Call6DofInteraction::IDX_GET_STATE = 0;


/*
 * megamol::core::view::Call6DofInteraction::Call6DofInteraction
 */
megamol::core::view::Call6DofInteraction::Call6DofInteraction(void) : buttonStates(0) {}


/*
 * megamol::core::view::Call6DofInteraction::~Call6DofInteraction
 */
megamol::core::view::Call6DofInteraction::~Call6DofInteraction(void) {}


/*
 * megamol::core::view::Call6DofInteraction::IsValid
 */
bool megamol::core::view::Call6DofInteraction::IsValid(void) const {
    bool retval = ((this->orientation.X() != 0) || (this->orientation.Y() != 0) || (this->orientation.Z() != 0) ||
                   (this->orientation.W() != 0) || (this->position.X() != 0) || (this->position.Y() != 0) ||
                   (this->position.Z() != 0));
    return retval;
}


/*
 * megamol::core::view::Call6DofInteraction::SetButtonState
 */
void megamol::core::view::Call6DofInteraction::SetButtonState(const int button, const bool isDown) {
    ButtonMaskType mask = (1 << button);
    if (isDown) {
        this->buttonStates |= mask;
    } else {
        this->buttonStates &= ~mask;
    }
}


/*
 * megamol::core::view::Call6DofInteraction::INTENTS
 */
const std::array<std::string, 1> megamol::core::view::Call6DofInteraction::INTENTS = {"getState"};
