#include "stdafx.h"
#include "SplineTubes.h"
#include <cassert>
#include <algorithm>

using namespace megamol;
using namespace megamol::beztube;
using namespace megamol::beztube::salm;

SplineTubes::SplineTubes() {
}

SplineTubes::~SplineTubes() {
}

void SplineTubes::allocate(ShaderBuffer::BufferType shaderBufferType, GLenum bufferUsageHintType, int bufferMaxElementCount, int *nodeCount, int segmentElementCount) {
    assert(bufferElementSize == 1 || bufferElementSize == 2 || bufferElementSize == 4 || bufferElementSize == 8);
    // assert(nodeCount.Length > 0);

    if (shaderBufferType == ShaderBuffer::UBO) {
        int maxBlockSize;
        glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxBlockSize);
        maxBlockSize /= (sizeof(float) * bufferElementSize);
        if (bufferMaxElementCount > 0)
            bufferMaxElementCount = std::min<int>(bufferMaxElementCount, maxBlockSize);
        else
            bufferMaxElementCount = maxBlockSize;
    } else {
        int maxBlockSize;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlockSize);
        maxBlockSize /= (sizeof(float) * bufferElementSize);
        if (bufferMaxElementCount > 0)
            bufferMaxElementCount = std::min<int>(bufferMaxElementCount, maxBlockSize);
        else
            bufferMaxElementCount = maxBlockSize;
    }

    this->bufferMaxElementCount = bufferMaxElementCount;
    this->bufferElementSize = bufferElementSize;

    this->nodeElementCount = /*TubeNode.Size*/ (4 * 4) / bufferElementSize;
    this->segmentElementCount = segmentElementCount;

    assert(bufferMaxElementCount >= (nodeElementCount + segmentElementCount) * 2);
    shaderBuffer.Allocate(shaderBufferType, bufferMaxElementCount * sizeof(float) * bufferElementSize, bufferUsageHintType);

    // This code is for data update
    /*
    this.tubeCount = nodeCount.Length;

    var nodeCountCapacity = new int[tubeCount];

    this.cummNodeCount = new int[tubeCount + 1];
    this.nodeCount = new int[tubeCount];

    for (int i = 0; i < tubeCount; ++i)
    {
        Utils.Assert(nodeCount[i] > 1,
            "nodeCount[i] > 1");

        cummNodeCount[i] = totalNodeCount;
        totalNodeCount += nodeCount[i];

        nodeCountCapacity[i] = nodeCount[i];
        this.nodeCount[i] = nodeCount[i];
    }

    cummNodeCount[tubeCount] = totalNodeCount;

    var totalNodeCountCapacity = totalNodeCount;

    var tubeId = 0;
    var bufferId = 0;

    var bufferCapacity = bufferMaxElementCount;
    var currSegmentCount = 0;
    var currNodeCount = 0;
    var maxBufferSize = 0;
    var dublicatedNode = false;

    while (totalNodeCountCapacity > 0)
    {
        var currNodeCountCapacity = nodeCountCapacity[tubeId];

        var firstNode = currNodeCountCapacity == nodeCount[tubeId];
        var lastNode = currNodeCountCapacity == 1;
        var secondlastNode = currNodeCountCapacity == 2;

        var currElementSize = nodeElementCount;
        var nextElementSize = nodeElementCount;

        if (!lastNode)
            currElementSize += segmentElementCount;

        if (!secondlastNode)
            nextElementSize += segmentElementCount;

        var currFits = bufferCapacity >= currElementSize;
        var nextFits = bufferCapacity >= currElementSize + nextElementSize;

        var addNewNode = currFits && (!firstNode || nextFits);

        if (addNewNode)
        {
            bufferCapacity -= currElementSize;

            if (nextFits || lastNode) {
                NodeMappingData mapping;
                mapping.BufferId = bufferId;
                mapping.Offset = currNodeCount * nodeElementCount;// not finalized
                mapping.SegmentOffset = -1;
                mapping.Dublicated = dublicatedNode;
                mapping.EndNode = lastNode;
                mapping.StartNode = firstNode;

                idToMemMap.Add(mapping);

                if (!lastNode)
                    ++currSegmentCount;

                ++currNodeCount;

                --totalNodeCountCapacity;
                --nodeCountCapacity[tubeId];

                if (lastNode) ++tubeId;

                dublicatedNode = false;
            } else
                dublicatedNode = true;
        }

        if (!addNewNode || totalNodeCountCapacity == 0) {
            var bufferSize = bufferMaxElementCount - bufferCapacity;

            Utils.Assert(bufferSize != 0,
                "bufferSize != 0");

            tubesBufferContent.Add(new ShaderBufferContent(bufferSize));
            segmentCount.Add(currSegmentCount);

            if (bufferSize > maxBufferSize) maxBufferSize = bufferSize;

            bufferCapacity = bufferMaxElementCount;
            currSegmentCount = 0;
            currNodeCount = 0;

            ++bufferId;
        }
    }

    shaderBufferCount = tubesBufferContent.Count;

    var bId = -1;
    var bOff = 0;
    var addOffset = 0;
    for (var nId = 0; nId < totalNodeCount; ++nId)
    {
        var mapping = idToMemMap[nId];

        if (mapping.BufferId > bId)
        {
            bId = mapping.BufferId;
            bOff = 0;

            addOffset = segmentCount[bId] * segmentElementCount;
        }

        mapping.Offset += addOffset;
        mapping.SegmentOffset = bOff;

        idToMemMap[nId] = mapping;

        if (!mapping.EndNode)
        {
            tubesBufferContent[bId].Fill(bOff, 0, (float)mapping.Offset);

            bOff += segmentElementCount;
        }
    }

    */
}
