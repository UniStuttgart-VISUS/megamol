/*
 * PoreMeshProcessor.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "PoreMeshProcessor.h"
//#include "vislib/Array.h"
#include "mmcore/utility/log/Log.h"
//#include "vislib/math/Point.h"
#include "vislib/sys/Thread.h"
//#include "vislib/sys/MemmappedFile.h"
//#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowPoint.h"

namespace megamol {
namespace demos_gl {

/*
 * PoreMeshProcessor::PoreMeshProcessor
 */
PoreMeshProcessor::PoreMeshProcessor() : vislib::sys::Runnable(), inputBuffers(NULL), debugoutschlupp(NULL) {
    // TODO: Implement
}


/*
 * PoreMeshProcessor::~PoreMeshProcessor
 */
PoreMeshProcessor::~PoreMeshProcessor() {
    // TODO: Implement
}


/*
 * PoreMeshProcessor::Run
 */
DWORD PoreMeshProcessor::Run(void* userData) {
    using megamol::core::utility::log::Log;
    ASSERT(this->inputBuffers != NULL);
    //ASSERT(this->outputBuffers != NULL);
    Log::DefaultLog.WriteInfo("PoreMeshProcessor Thread started");
    this->sliceNum = 0;

    // TODO: Make abstract base classes (producer, consumer, processor?)

    while (true) {
        vislib::sys::Thread::Sleep(1);
        LoopBuffer* inbuffer = this->inputBuffers->GetFilledBuffer(true);
        if (inbuffer == NULL) {
            if (this->inputBuffers->IsEndOfData()) {
                // TODO: Fixme
                // this->outputBuffers->EndOfDataClose();
                // graceful finishing line :-)
            }
            break;
        }
        //LoopBuffer *outbuffer = this->outputBuffers->GetEmptyBuffer(true);
        //if (outbuffer == NULL) {
        //    break;
        //}
        //outbuffer->Clear();

        this->workOnBuffer(*inbuffer /*, *outbuffer*/);

        this->inputBuffers->BufferConsumed(inbuffer);
        //this->outputBuffers->BufferFilled(outbuffer);
    }

    Log::DefaultLog.WriteInfo("PoreMeshProcessor Thread stopped");
    return 0;
}


/*
 * PoreMeshProcessor::Terminate
 */
bool PoreMeshProcessor::Terminate() {
    if (this->inputBuffers != NULL)
        this->inputBuffers->AbortClose();
    //if (this->outputBuffers != NULL) this->outputBuffers->AbortClose();
    return true;
}


/*
 * PoreMeshProcessor::workOnBuffer
 */
void PoreMeshProcessor::workOnBuffer(LoopBuffer& buffer /*, LoopBuffer& outBuffer*/) {

    // TODO: Implement something more useful

    SliceLoops* sd = new SliceLoops();
    sd->cnt = 0;
    sd->data = NULL;
    sd->next = NULL;

    for (SIZE_T i = 0; i < buffer.Loops().Count(); i++) {
        sd->cnt += 2 * static_cast<unsigned int>(buffer.Loops()[i].Length());
    }
    sd->data = new float[sd->cnt * 3];
    sd->cnt = 0;
    for (SIZE_T i = 0; i < buffer.Loops().Count(); i++) {
        SIZE_T len = buffer.Loops()[i].Length();
        for (SIZE_T j = 0; j < len; j++) {
            vislib::math::ShallowPoint<float, 3> v1(sd->data + (sd->cnt * 3));
            vislib::math::ShallowPoint<float, 3> v2(sd->data + ((sd->cnt + 1) * 3));
            const vislib::math::Point<int, 2>& p1(buffer.Loops()[i].Vertex(j));
            const vislib::math::Point<int, 2>& p2(buffer.Loops()[i].Vertex((j + 1) % len));
            sd->cnt += 2;

            v1 = this->origin;
            v1 += this->axes[0] * static_cast<float>(p1.X());
            v1 += this->axes[1] * static_cast<float>(p1.Y());
            v1 += this->axes[2] * (static_cast<float>(this->sliceNum) + 0.5f);

            v2 = this->origin;
            v2 += this->axes[0] * static_cast<float>(p2.X());
            v2 += this->axes[1] * static_cast<float>(p2.Y());
            v2 += this->axes[2] * (static_cast<float>(this->sliceNum) + 0.5f);
        }
    }

    SliceLoops* l = this->debugoutschlupp;
    while (l->next != NULL)
        l = l->next;
    l->next = sd;

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Slice %u added to loop-debug-data\n", this->sliceNum);

    this->sliceNum++;
}

} // namespace demos_gl
} /* end namespace megamol */
