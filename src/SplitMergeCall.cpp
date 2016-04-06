#include "stdafx.h"
#include "SplitMergeCall.h"

namespace megamol {
namespace protein_cuda {

/*
 * SplitMergeCall::CallForGetData
 */
const unsigned int protein_cuda::SplitMergeCall::CallForGetData = 0;

/*
 * SplitMergeCall::SplitMergeCall
 */
SplitMergeCall::SplitMergeCall(void) : theData(), theTransitions(), guides() {
    // set reasonable resize and initial capacity for data arrays
    this->theData.AssertCapacity( 1000);
    this->theData.SetCapacityIncrement( 100);
    this->guides.AssertCapacity( 1000);
    this->guides.SetCapacityIncrement( 100);
}


/*
 * SplitMergeCall::~SplitMergeCall
 */
SplitMergeCall::~SplitMergeCall(void) {

}


/*
 * SplitMergeTransition::SplitMergeTransition
 */
SplitMergeCall::SplitMergeTransition::SplitMergeTransition( 
    unsigned int sourceSeries, unsigned int sourceSeriesIdx, float sourceWidth, 
    unsigned int destinationSeries, unsigned int destinationSeriesIdx, float destinationWidth) : 
        visible(true) {
    // source and destination series may not be the same
    ASSERT( sourceSeries != destinationSeries);
    // ensure that the source series is smaller than the destination series
    if( sourceSeries < destinationSeries ) {
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

} /* end namespace protein_cuda */
} /* end namespace megamol */
