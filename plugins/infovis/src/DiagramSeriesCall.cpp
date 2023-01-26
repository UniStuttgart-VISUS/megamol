#include "DiagramSeriesCall.h"

const unsigned int megamol::infovis::DiagramSeriesCall::IdIdx = 0;

const unsigned int megamol::infovis::DiagramSeriesCall::ColIdx = 1;

const unsigned int megamol::infovis::DiagramSeriesCall::NameIdx = 2;

const unsigned int megamol::infovis::DiagramSeriesCall::ScalingIdx = 3;

const unsigned int megamol::infovis::DiagramSeriesCall::CallForGetSeries = 0;

megamol::infovis::DiagramSeriesCall::DiagramSeriesCall() : core::Call() {}

megamol::infovis::DiagramSeriesCall::~DiagramSeriesCall() {}

megamol::infovis::DiagramSeriesCall& megamol::infovis::DiagramSeriesCall::operator=(const DiagramSeriesCall& rhs) {
    this->ptmSeriesInsertionCB = rhs.ptmSeriesInsertionCB;
    return *this;
}

megamol::infovis::DiagramSeriesCall::fpSeriesInsertionCB
megamol::infovis::DiagramSeriesCall::GetSeriesInsertionCB() const {
    return this->ptmSeriesInsertionCB;
}

void megamol::infovis::DiagramSeriesCall::SetSeriesInsertionCB(const fpSeriesInsertionCB& fpsicb) {
    this->ptmSeriesInsertionCB = fpsicb;
}
