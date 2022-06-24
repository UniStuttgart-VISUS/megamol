#include "protein_calls/SplitMergeCall.h"

namespace megamol {
namespace protein_calls {

/*
 * SplitMergeCall::CallForGetData
 */
const unsigned int SplitMergeCall::CallForGetData = 0;

/*
 * SplitMergeCall::SplitMergeCall
 */
SplitMergeCall::SplitMergeCall(void) : theTransitions(), guides() {
    // set reasonable resize and initial capacity for data arrays
    this->theData = new vislib::Array<SplitMergeSeries*>();
    this->guides = new vislib::PtrArray<SplitMergeGuide>();

    this->theData->AssertCapacity(1000);
    this->theData->SetCapacityIncrement(100);
    this->guides->AssertCapacity(1000);
    this->guides->SetCapacityIncrement(100);
}


/*
 * SplitMergeCall::~SplitMergeCall
 */
SplitMergeCall::~SplitMergeCall(void) {
    delete this->theData;
    delete this->guides;
}


/*
 * SplitMergeTransition::SplitMergeTransition
 */
SplitMergeCall::SplitMergeTransition::SplitMergeTransition(unsigned int sourceSeries, unsigned int sourceSeriesIdx,
    float sourceWidth, unsigned int destinationSeries, unsigned int destinationSeriesIdx, float destinationWidth)
        : visible(true) {
    // source and destination series may not be the same
    ASSERT(sourceSeries != destinationSeries);
    // ensure that the source series is smaller than the destination series
    if (sourceSeries < destinationSeries) {
        srcSeries = sourceSeries;
        srcSeriesIdx = sourceSeriesIdx;
        srcWidth = sourceWidth;
        dstSeries = destinationSeries;
        dstSeriesIdx = destinationSeriesIdx;
        dstWidth = destinationWidth;
    } else {
        srcSeries = destinationSeries;
        srcSeriesIdx = destinationSeriesIdx;
        srcWidth = destinationWidth;
        dstSeries = sourceSeries;
        dstSeriesIdx = sourceSeriesIdx;
        dstWidth = sourceWidth;
    }
}

} /* end namespace protein_calls */
} /* end namespace megamol */
