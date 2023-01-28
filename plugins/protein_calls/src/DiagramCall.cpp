#include "protein_calls/DiagramCall.h"

namespace megamol::protein_calls {

/*
 * MolecularDataCall::CallForGetData
 */
const unsigned int DiagramCall::CallForGetData = 0;

/*
 * Diagram2DCall::Diagram2DCall
 */
DiagramCall::DiagramCall() {
    this->theData = new vislib::Array<DiagramSeries*>();
    this->guides = new vislib::PtrArray<DiagramGuide>();
}


/*
 * Diagram2DCall::~Diagram2DCall
 */
DiagramCall::~DiagramCall() {
    delete this->theData;
    delete this->guides;
}

} // namespace megamol::protein_calls
