#include "MSMDataCall.h"

megamol::archvis::MSMDataCall::MSMDataCall() : AbstractGetDataCall(), m_MSM(nullptr) {}

void megamol::archvis::MSMDataCall::setMSM(std::shared_ptr<ScaleModel> const& msm) {
    m_MSM = msm;
}

std::shared_ptr<megamol::archvis::ScaleModel> megamol::archvis::MSMDataCall::getMSM() {
    return m_MSM;
}
