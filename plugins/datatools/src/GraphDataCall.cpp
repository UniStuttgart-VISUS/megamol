#include "datatools/GraphDataCall.h"

using namespace megamol;
using namespace megamol::datatools;

GraphDataCall::GraphDataCall()
        : core::AbstractGetDataCall()
        , edge_count(0)
        , edge_data(nullptr)
        , is_directed(false)
        , frameCnt(1)
        , frameID(0) {
    // intentionally empty
}

GraphDataCall::~GraphDataCall() {
    edge_count = 0; // paranoia
    edge_data = nullptr;
}
