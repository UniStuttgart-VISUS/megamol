/*
 * PoreNetSliceProcessor.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

//#define DEBUG_BMP
#include "PoreNetSliceProcessor.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/Array.h"
#include "vislib/math/Point.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/sys/Thread.h"
#ifdef DEBUG_BMP
#include "vislib/graphics/BitmapImage.h"
#include "vislib/graphics/BitmapPainter.h"
#include "vislib/graphics/BmpBitmapCodec.h"
#endif /* DEBUG_BMP */

namespace megamol {
namespace demos_gl {

static const char rightmostPoint[16] = {-1, 1, 2, 2, 3, -1, 3, 3, 0, 1, -1, 2, 0, 1, 0, -1};
static const char rightmostEdge[16] = {-1, 1, 2, 2, 4, -1, 4, 4, 8, 1, -1, 2, 8, 1, 8, -1};

static const char leftmostPoint[16] = {-1, 0, 1, 0, 2, -1, 1, 0, 3, 3, -1, 3, 2, 2, 1, -1};
static const char leftmostEdge[16] = {-1, 1, 2, 1, 4, -1, 2, 1, 8, 8, -1, 8, 4, 4, 2, -1};


/*
 * PoreNetSliceProcessor::PoreNetSliceProcessor
 */
PoreNetSliceProcessor::PoreNetSliceProcessor(void) : vislib::sys::Runnable(), inputBuffers(NULL), outputBuffers(NULL) {
    // TODO: Implement
    dirOffset[0].Set(0, -1);
    dirOffset[1].Set(1, 0);
    dirOffset[2].Set(0, 1);
    dirOffset[3].Set(-1, 0);

    edgeOffset[0].Set(0, 0);
    edgeOffset[1].Set(0, -1);
    edgeOffset[2].Set(1, 0);
    edgeOffset[3].Set(0, 0);
    edgeOffset[4].Set(0, 1);
    edgeOffset[5].Set(0, 0);
    edgeOffset[6].Set(0, 0);
    edgeOffset[7].Set(0, 0);
    edgeOffset[8].Set(-1, 0);
}


/*
 * PoreNetSliceProcessor::~PoreNetSliceProcessor
 */
PoreNetSliceProcessor::~PoreNetSliceProcessor(void) {
    // TODO: Implement
}


/*
 * PoreNetSliceProcessor::Run
 */
DWORD PoreNetSliceProcessor::Run(void* userData) {
    using megamol::core::utility::log::Log;
    ASSERT(this->inputBuffers != NULL);
    ASSERT(this->outputBuffers != NULL);
    Log::DefaultLog.WriteInfo("PoreNetSliceProcessing Thread started");

    // TODO: Make abstract base classes (producer, consumer, processor?)

    while (true) {
        vislib::sys::Thread::Sleep(1);
        ArxelBuffer* inbuffer = this->inputBuffers->GetFilledBuffer(true);
        if (inbuffer == NULL) {
            if (this->inputBuffers->IsEndOfData()) {
                this->outputBuffers->EndOfDataClose();
                // graceful finishing line :-)
            }
            break;
        }
        LoopBuffer* outbuffer = this->outputBuffers->GetEmptyBuffer(true);
        if (outbuffer == NULL) {
            break;
        }
        outbuffer->Clear();

        this->workOnBuffer(*inbuffer, *outbuffer);

        this->inputBuffers->BufferConsumed(inbuffer);
        this->outputBuffers->BufferFilled(outbuffer);
    }

    Log::DefaultLog.WriteInfo("PoreNetSliceProcessing Thread stopped");
    return 0;
}


/*
 * PoreNetSliceProcessor::Terminate
 */
bool PoreNetSliceProcessor::Terminate(void) {
    if (this->inputBuffers != NULL)
        this->inputBuffers->AbortClose();
    if (this->outputBuffers != NULL)
        this->outputBuffers->AbortClose();
    return true;
}


/*
 * PoreNetSliceProcessor::collectEdge
 */
void PoreNetSliceProcessor::collectEdge(
    LoopBuffer& outBuffer, ArxelBuffer& buffer, ArxelBuffer& edgeStore, int x, int y, cellDirection inDir) {

    // is there something in direction inDir
    char currEdgeIdx = 1 << inDir;
    char rightEdgeIdx;
    char currEdgeFlags = (edgeStore.Get(x, y));
    if ((currEdgeFlags & currEdgeIdx) == 0) {
        return;
    }
    BorderElement currCell, startCell, firstCell, lastCell;
    //vislib::Array<vislib::math::Point<int, 2> > currStrip(10,20);
    LoopBuffer::Loop& currLoop(outBuffer.NewLoop());

    firstCell.edges = 0;
    lastCell.edges = currEdgeIdx;
    startCell.pos.Set(x, y);
    currCell.pos.Set(x, y);
    lastCell.pos.Set(x, y);
    vislib::math::Point<int, 2> nextPos(0, 0);
    int ww = 0, w = 0, v = buffer.Get((currCell.pos + edgeOffset[currEdgeIdx]));
    char nextDir, dir = inDir;
    char left, leftEdgeFlags, right, frontLeftEdgeFlags;

    do {
        currCell.edges = 0;
        currEdgeFlags = edgeStore.Get(currCell.pos);
        // search cell-local connectivity
        if ((currEdgeFlags & currEdgeIdx) == 0) {
            break;
        }
        while ((currEdgeFlags & currEdgeIdx) != 0) {
            w = buffer.Get((currCell.pos + edgeOffset[currEdgeIdx]));
            if (w == v) {
                currCell.edges |= currEdgeIdx;
                currCell.val = w;
                // consume edge
                currEdgeFlags -= currEdgeIdx;
                // turn left
                dir = (dir - 1 + 4) % 4;
                currEdgeIdx = 1 << dir;
                lastCell = currCell;
                w = buffer.Get((currCell.pos + edgeOffset[currEdgeIdx]));
            } else {
                break;
            }
        }
        // color changed, take back turn
        dir = (dir + 1) % 4;
        currEdgeIdx = 1 << dir;
        ww = 0;
        // does the color continue in neighbors?
        // option 1: edge continues straight in left cell
        left = (dir - 1 + 4) % 4;
        leftEdgeFlags = edgeStore.Get(currCell.pos + dirOffset[left]);
        if ((leftEdgeFlags & currEdgeIdx) > 0) {
            ww = buffer.Get(currCell.pos + dirOffset[left] + edgeOffset[currEdgeIdx]);
            nextPos = currCell.pos + dirOffset[left];
            nextDir = dir;
            if (ww == v) {
                edgeStore.Set(currCell.pos, currEdgeFlags);
                // straight
                currCell.pos = nextPos;
                // dir and EdgeIdx unchanged!
                continue;
            }
        } else {
            // option 2: 1 forward, 1 left has edge +90 deg
            frontLeftEdgeFlags = edgeStore.Get(currCell.pos + dirOffset[dir] + dirOffset[left]);
            right = (dir + 1) % 4;
            rightEdgeIdx = 1 << right;
            if ((frontLeftEdgeFlags & rightEdgeIdx) > 0) {
                ww = buffer.Get(currCell.pos + dirOffset[dir] + dirOffset[left] + edgeOffset[rightEdgeIdx]);
                nextPos = currCell.pos + dirOffset[dir] + dirOffset[left];
                nextDir = right;
                if (ww == v) {
                    edgeStore.Set(currCell.pos, currEdgeFlags);
                    // corner in frontleft
                    currCell.pos = nextPos;
                    dir = right;
                    currEdgeIdx = 1 << right;
                    continue;
                } else {
                    // we ran out ZOMG
                    edgeStore.Set(currCell.pos, currEdgeFlags);
                    break;
                }
            }
        }
        // we must accept to change color, sigh
        // was there a colored edge left in this cell?
        if (w > 0) {
            edgeStore.Set(currCell.pos, currEdgeFlags);
            // turn left
            dir = (dir - 1 + 4) % 4;
            currEdgeIdx = 1 << dir;
            v = w;
        } else if (ww > 0) {
            // the step to the neighboring cell would have been a better idea huh!
            v = ww;
            edgeStore.Set(currCell.pos, currEdgeFlags);
            currCell.pos = nextPos;
            dir = nextDir;
            currEdgeIdx = 1 << dir;
        } else {
            // we ran out ZOMG
            edgeStore.Set(currCell.pos, currEdgeFlags);
            break;
        }
        if (firstCell.edges == 0) {
            // this is our first color change
            firstCell = lastCell;
            //currStrip.Add(currCell.pos + edgeOffset[currEdgeIdx]);
            //addToStrip(buffer, currStrip, currCell.pos, edgeOffset[currEdgeIdx]);

            vislib::math::Point<int, 2> p1 = firstCell.pos; // + edgeOffset[leftmostEdge[firstCell.edges]];
            vislib::math::Point<int, 2> p2 = currCell.pos;  // + edgeOffset[currEdgeIdx];

            int x = p1.X() + p2.X();
            if ((x % 2) == 1)
                x = x / 2 + 1;
            else
                x /= 2;
            int y = p1.Y() + p2.Y();
            if ((y % 2) == 1)
                y = y / 2 + 1;
            else
                y /= 2;
            currLoop.AddVertex(vislib::math::Point<int, 2>(x, y), currCell.val);

        } else {
            //currStrip.Add(lastCell.pos + edgeOffset[leftmostEdge[lastCell.edges]]);
            //addToStrip(buffer, currStrip, lastCell.pos, edgeOffset[leftmostEdge[lastCell.edges]]);
            ////currStrip.Add(currCell.pos + edgeOffset[currEdgeIdx]);
            //addToStrip(buffer, currStrip, currCell.pos, edgeOffset[currEdgeIdx]);

            vislib::math::Point<int, 2> p1 = lastCell.pos; // + edgeOffset[leftmostEdge[lastCell.edges]];
            vislib::math::Point<int, 2> p2 = currCell.pos; // + edgeOffset[currEdgeIdx];

            int x = p1.X() + p2.X();
            if ((x % 2) == 1)
                x = x / 2 + 1;
            else
                x /= 2;
            int y = p1.Y() + p2.Y();
            if ((y % 2) == 1)
                y = y / 2 + 1;
            else
                y /= 2;
            currLoop.AddVertex(vislib::math::Point<int, 2>(x, y), currCell.val);
        }
    } while ((currCell.pos != startCell.pos) || (dir != inDir));

    //currLoop.SetArea(buffer.Fill(currLoop.Vertices(), 255/*, true*/));
    currLoop.SetArea(buffer.Fill(currLoop.Vertices(), 0, true));
    outBuffer.NewLoopComplete();

    //currStrip.Add(firstCell.pos + edgeOffset[leftmostEdge[firstCell.edges]]);
    //addToStrip(buffer, currStrip, firstCell.pos, edgeOffset[leftmostEdge[firstCell.edges]]);
    //ASSERT(currStrip.Count() % 2 == 0);
    //strips.Add(currStrip); // ???
}


/*
 * PoreNetSliceProcessor::isEdgePixel
 */
bool PoreNetSliceProcessor::isEdgePixel(ArxelBuffer& buffer, int x, int y) {
    return (buffer.Get(x - 1, y - 1) == 0 || buffer.Get(x, y - 1) == 0 || buffer.Get(x + 1, y - 1) == 0 ||
            buffer.Get(x - 1, y) == 0 || buffer.Get(x + 1, y) == 0 || buffer.Get(x - 1, y + 1) == 0 ||
            buffer.Get(x, y + 1) == 0 || buffer.Get(x + 1, y + 1) == 0);
}

/*
 * PoreNetSliceProcessor::workOnBuffer
 */
void PoreNetSliceProcessor::workOnBuffer(ArxelBuffer& buffer, LoopBuffer& outBuffer) {

#ifdef DEBUG_BMP
    vislib::graphics::BitmapImage bmp(buffer.Width(), buffer.Height(), vislib::graphics::BitmapImage::TemplateByteRGB);
    vislib::graphics::BitmapPainter bmpDraw(&bmp);
    for (unsigned int x = 0; x < buffer.Width(); x++) {
        for (unsigned int y = 0; y < buffer.Height(); y++) {
            ArxelBuffer::ArxelType val = buffer.Get(x, y);
            bmpDraw.SetColour<BYTE, ArxelBuffer::ArxelType, ArxelBuffer::ArxelType>(
                (val == static_cast<ArxelBuffer::ArxelType>(0)) ? 0 : 255, val, val);
            bmpDraw.SetPixel(x, y);
        }
    }
#endif /* DEBUG_BMP */

    //#define DEBUG_SLICE_FILE_WRITE
#ifdef DEBUG_SLICE_FILE_WRITE
    vislib::sys::MemmappedFile debugFile;
    debugFile.Open("C:\\temp\\bananenwurst.dat", vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_READ,
        vislib::sys::File::CREATE_OVERWRITE);
#endif /* DEBUG_SLICE_FILE_WRITE */

    // DEBUG
//#define WTF_READ_SIMPLIFIED
#ifdef WTF_READ_SIMPLIFIED
    vislib::sys::MemmappedFile* inf = new vislib::sys::MemmappedFile();
    if (!inf->Open("t:\\data\\inextract.dat", vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
            vislib::sys::File::OPEN_ONLY)) {
        SAFE_DELETE(inf);
    }
    if (inf != NULL) {
        inf->Read(buffer.Data(), buffer.Width() * buffer.Height() * sizeof(ArxelBuffer::ArxelType));
        //this->outFile->Write(buffer.Data(), buffer.Width() * buffer.Height() * sizeof(ArxelBuffer::ArxelType));
        inf->Close();
        SAFE_DELETE(inf);
    }
#endif /* WTF_READ_SIMPLIFIED */

    // Step.1 Remove isolated arxels
    unsigned int isolatedHoles = 0;
    vislib::Array<vislib::math::Point<int, 2>> isolatedNotHoles;

#ifdef DEBUG_BMP
    bmpDraw.SetColour<BYTE, BYTE, BYTE>(128, 128, 128);
#endif /* DEBUG_BMP */
    for (int y = 0; y < static_cast<int>(buffer.Height()); y++) {
        for (int x = 0; x < static_cast<int>(buffer.Width()); x++) {
            ArxelBuffer::ArxelType p = buffer.Get(x, y);
            ArxelBuffer::ArxelType n[8] = {buffer.Get(x - 1, y - 1), buffer.Get(x, y - 1), buffer.Get(x + 1, y - 1),
                buffer.Get(x - 1, y), buffer.Get(x + 1, y), buffer.Get(x - 1, y + 1), buffer.Get(x, y + 1),
                buffer.Get(x + 1, y + 1)};
            if ((p != n[0]) && (p != n[1]) && (p != n[2]) && (p != n[3]) && (p != n[4]) && (p != n[5]) && (p != n[6]) &&
                (p != n[7])) {
                // p is isolated arxel

                ArxelBuffer::ArxelType pnew = p;
                unsigned int mc = 0, c;
                for (unsigned int i = 0; i < 8; i++) {
                    c = 0;
                    for (unsigned int j = 0; j < 8; j++) {
                        if (n[i] == n[j])
                            c++;
                    }
                    if (c > mc) {
                        mc = c;
                        pnew = n[i];
                    }
                }

                if (p == 0) {
                    ASSERT(pnew != 0);
#ifdef DEBUG_BMP
                    bmpDraw.SetPixel(x, y);
#endif /* DEBUG_BMP */
                    isolatedHoles++;
                } else if (pnew == 0) { /** isolatedNotHoles are only isolated when surrounded by hole! */
                    isolatedNotHoles.Add(vislib::math::Point<int, 2>(x, y));
                }

                buffer.Set(x, y, pnew);
            }
        }
    }
    outBuffer.SetBlackArxels(isolatedHoles);
    outBuffer.Bounds().Set(buffer.Width(), buffer.Height());

#ifdef DEBUG_SLICE_FILE_WRITE
    if (debugFile.IsOpen()) {
        debugFile.Write(buffer.Data(), buffer.Width() * buffer.Height() * sizeof(ArxelBuffer::ArxelType));
    }
#endif /* DEBUG_SLICE_FILE_WRITE */

    ArxelBuffer edgeStore;
    ArxelBuffer::InitValues abiv;
    abiv.width = buffer.Width();
    abiv.height = buffer.Height();
    int state;
    ArxelBuffer::Initialize(edgeStore, state, abiv);
    edgeStore.SetBorders(0, 0);
    for (int y = 0; y < static_cast<int>(abiv.height); y++) {
        for (int x = 0; x < static_cast<int>(abiv.width); x++) {
            ArxelBuffer::ArxelType cur = 0; // = edgeStore.Get(x, y);
            if (buffer.Get(x, y) == 0) {
                if (buffer.Get(x - 1, y) != 0) { // || x == 0) {
                    cur |= 8;
                }
                if (buffer.Get(x + 1, y) != 0) { // || x == abiv.width - 1) {
                    cur |= 2;
                }
                if (buffer.Get(x, y - 1) != 0) { // || y == 0) {
                    cur |= 1;
                }
                if (buffer.Get(x, y + 1) != 0) { // || y == abiv.height - 1) {
                    cur |= 4;
                }
            }
            edgeStore.Set(x, y, cur);
        }
    }


    // extract the vertices

    vislib::Array<vislib::Array<vislib::math::Point<int, 2>>> strips;

    vislib::Array<vislib::math::Point<int, 2>> vertices(10, 100);
    for (int y = 0; y <= static_cast<int>(buffer.Height()); y++) {
        for (int x = 0; x <= static_cast<int>(buffer.Width()); x++) {
            if (buffer.Get(x, y) == 0) {
                collectEdge(outBuffer, /*strips, */ buffer, edgeStore, x, y, Right);
                collectEdge(outBuffer, /*strips, */ buffer, edgeStore, x, y, Down);
                collectEdge(outBuffer, /*strips, */ buffer, edgeStore, x, y, Left);
                collectEdge(outBuffer, /*strips, */ buffer, edgeStore, x, y, Up);
            }
        }
    }

    for (int y = 0; y < static_cast<int>(buffer.Height()); y++) {
        for (int x = 0; x < static_cast<int>(buffer.Width()); x++) {
            if (buffer.Get(x, y) != 0)
                buffer.Set(x, y, 0);
        }
    }

    // loop containment info generating code phun
    for (SIZE_T i = 0; i < outBuffer.Loops().Count(); i++) {
        LoopBuffer::Loop& li = outBuffer.Loops()[i];
        const LoopBuffer::Loop* el = NULL;
        int maxLeftX = -1;
        ASSERT(li.Length() > 0);
        int lix = li.Vertex(0).X();
        int liy = li.Vertex(0).Y();

        for (SIZE_T j = 0; j < outBuffer.Loops().Count(); j++) {
            if (i == j)
                continue;
            const LoopBuffer::Loop& lj = outBuffer.Loops()[j];
            if (!lj.BoundingBox().Contains(li.Vertex(0)))
                continue;

            unsigned int leftEdgeCnt = 0; // number of edges left of li.V(0)
            int lmlx = -1;

            for (SIZE_T k = 0; k < lj.Length(); k++) {
                const vislib::math::Point<int, 2>& p = lj.Vertex(k);
                const vislib::math::Point<int, 2>& pn = lj.Vertex((k + 1) % lj.Length());
                int yMin = vislib::math::Min(p.Y(), pn.Y());
                int yMax = vislib::math::Max(p.Y(), pn.Y()); // no part of edge

                if ((yMin > liy) || (yMax <= liy))
                    continue;

                int x = static_cast<int>(static_cast<float>(p.X()) + static_cast<float>(pn.X() - p.X()) *
                                                                         static_cast<float>(liy - p.Y()) /
                                                                         static_cast<float>(pn.Y() - p.Y()));

                if (x < lix) {
                    leftEdgeCnt++;
                    if (x > lmlx)
                        lmlx = x;
                }
            }

            if (((leftEdgeCnt % 2) == 1) && (lmlx > maxLeftX)) {
                maxLeftX = lmlx;
                el = &lj;
            }
        }

        li.SetEnclosingLoop(el);
    }

    // sort loops by nesticity
    vislib::Array<vislib::Pair<SIZE_T, unsigned int>> loopIdx;
    for (SIZE_T i = 0; i < outBuffer.Loops().Count(); i++) {
        LoopBuffer::Loop& li = outBuffer.Loops()[i];
        const LoopBuffer::Loop* el = li.EnclosingLoop();
        int count = 0;
        while (el) {
            count++;
            el = el->EnclosingLoop();
        }
        loopIdx.Add(vislib::Pair<SIZE_T, unsigned int>(i, count));
    }
    loopIdx.Sort(vislib::math::ComparePairsSecond);

    // fix area values: pores without nested stuff, matrix-loops to zero
    for (SIZE_T i = 0; i < loopIdx.Count(); i++) {
        LoopBuffer::Loop& loop = outBuffer.Loops()[loopIdx[i].First()];
        if ((loopIdx[i].Second() % 2) == 1) {
            ASSERT(loop.EnclosingLoop() != NULL);
            const_cast<LoopBuffer::Loop*>(loop.EnclosingLoop())->SetArea(loop.EnclosingLoop()->Area() - loop.Area());
            loop.SetArea(0);
        } else {
            ASSERT((loop.EnclosingLoop() == NULL) || (loop.EnclosingLoop()->Area() == 0));
        }
    }

    // assign the isolated non-blacks to the smallest enclosing loop
    unsigned int orphanedWhites = 0;
#ifdef DEBUG_BMP
    bmpDraw.SetColour<BYTE, BYTE, BYTE>(255, 128, 0);
#endif /* DEBUG_BMP */
    for (SIZE_T i = 0; i < isolatedNotHoles.Count(); i++) {
        vislib::math::Point<int, 2>& p = isolatedNotHoles[i];
        bool consumed = false;
        for (SIZE_T j = loopIdx.Count(); j > 0; j--) {
            LoopBuffer::Loop& li = outBuffer.Loops()[loopIdx[j - 1].First()];
            if (li.Contains(p) /* && (li.Area() > 0)*/) {
                //ASSERT(li.Area() > 0);
                li.SetWhiteArxels(li.WhiteArxels() + 1);
                consumed = true;
                break;
            }
        }
        if (!consumed) {
#ifdef DEBUG_BMP
            bmpDraw.SetPixel(p.X(), p.Y());
#endif /* DEBUG_BMP */
            orphanedWhites++;
        }
    }
    if (orphanedWhites > 0) {
        // TODO: BugBugBugBugBugBugBugBugHeck
        outBuffer.SetBlackArxels(outBuffer.BlackArxels() - orphanedWhites);
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Found %u orphaned non-black Arxels.", orphanedWhites);
    }
#ifdef DEBUG_SLICE_FILE_WRITE
    // first the holes...
    for (SIZE_T x = 0; x < outBuffer.Loops().Count(); x++) {
        if (outBuffer.Loops()[x].Area() > 0) {
            buffer.Fill(outBuffer.Loops()[x].Vertices(), 96);
        }
    }
    // then the floating crystals
    for (SIZE_T x = 0; x < outBuffer.Loops().Count(); x++) {
        if (outBuffer.Loops()[x].Area() == 0) {
            buffer.Fill(outBuffer.Loops()[x].Vertices(), 255);
        }
    }
    // holes in floating crystals etc. exist but are obfuscated because guido is a lazy sack of the essence

    if (debugFile.IsOpen()) {
        debugFile.Write(buffer.Data(), buffer.Width() * buffer.Height() * sizeof(ArxelBuffer::ArxelType));
    }
#endif /* DEBUG_SLICE_FILE_WRITE */

#ifdef DEBUG_SLICE_FILE_WRITE
    if (debugFile.IsOpen()) {
        debugFile.Close();
    }
#endif /* DEBUG_SLICE_FILE_WRITE */

#ifdef DEBUG_BMP
    for (SIZE_T x = 0; x < outBuffer.Loops().Count(); x++) {
        if (outBuffer.Loops()[x].Area() > 0) {
            bmpDraw.SetColour<BYTE, BYTE, BYTE>(0, 0, 255);
        } else {
            bmpDraw.SetColour<BYTE, BYTE, BYTE>(0, 200, 0);
        }
        for (SIZE_T i = 0; i < outBuffer.Loops()[x].Length(); i++) {
            bmpDraw.DrawLine(outBuffer.Loops()[x].Vertex(i).X(), outBuffer.Loops()[x].Vertex(i).Y(),
                outBuffer.Loops()[x].Vertex((i + 1) % outBuffer.Loops()[x].Length()).X(),
                outBuffer.Loops()[x].Vertex((i + 1) % outBuffer.Loops()[x].Length()).Y());
        }
    }

    for (SIZE_T i = 0; i < isolatedNotHoles.Count(); i++) {
        vislib::math::Point<int, 2>& p = isolatedNotHoles[i];
        bool consumed = false;
        for (SIZE_T j = loopIdx.Count(); j > 0; j--) {
            LoopBuffer::Loop& li = outBuffer.Loops()[loopIdx[j - 1].First()];
            if (li.Contains(p) /* && (li.Area() > 0)*/) {
                //ASSERT(li.Area() > 0);
                consumed = true;
                break;
            }
        }
        if (!consumed) {
            bmpDraw.SetColour<BYTE, BYTE, BYTE>(255, 128, 0);
        } else {
            bmpDraw.SetColour<BYTE, BYTE, BYTE>(0, 255, 192);
        }
        bmpDraw.SetPixel(p.X(), p.Y());
    }

    vislib::graphics::BmpBitmapCodec codec;
    codec.Image() = &bmp;
    codec.Save("C:\\tmp\\bananenwurst.bmp");
#endif /* DEBUG_BMP */
}

} // namespace demos_gl
} /* end namespace megamol */
