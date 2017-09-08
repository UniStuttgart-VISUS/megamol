#pragma once

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "salm/ShaderBuffer.h"

namespace megamol {
namespace beztube {
namespace salm {

    class SplineTubes {
    private:
        int bufferElementSize = 4;
        int bufferMaxElementCount = 4096;

        //protected List<ShaderBufferContent> tubesBufferContent = new List<ShaderBufferContent>(1);
        //protected List<int> segmentCount = new List<int>(1);
        ShaderBuffer shaderBuffer;

        int nodeElementCount;
        int segmentElementCount;

        int tubeCount;
        int totalNodeCount = 0;
        int shaderBufferCount;

    public:
        SplineTubes();
        virtual ~SplineTubes();

    protected:

        void allocate(ShaderBuffer::BufferType shaderBufferType, GLenum bufferUsageHintType, int bufferMaxElementCount, int *nodeCount, int segmentElementCount);

#if 0

        public void FixZeroTangents(int tubeId)
        {
            Utils.Assert(tubeId >= 0 && tubeId < tubeCount,
                "tubeId >= 0 && tubeId < tubeCount");

            var baseNodeId = cummNodeCount[tubeId];
            var endNodeId = baseNodeId + nodeCount[tubeId] - 1;

            for (var nodeId = baseNodeId; nodeId < endNodeId; ++nodeId)
            {
                var mapping = idToMemMap[nodeId];

                var content = tubesBufferContent[mapping.BufferId];
                var data = content.Content;


                var off1 = mapping.Offset * 4;
                var off2 = off1 + nodeElementCount * 4;

                var pos1 = new Vector3(data[off1 + 0], data[off1 + 1], data[off1 + 2]);
                var tan1 = new Vector3(data[off1 + 4], data[off1 + 5], data[off1 + 6]);

                var pos2 = new Vector3(data[off2 + 0], data[off2 + 1], data[off2 + 2]);
                var tan2 = new Vector3(data[off2 + 4], data[off2 + 5], data[off2 + 6]);


                var tan1ZeroCond = tan1.X == 0.0 && tan1.Y == 0.0 && tan1.Z == 0.0;
                var tan2ZeroCond = tan2.X == 0.0 && tan2.Y == 0.0 && tan2.Z == 0.0;

                var small = 0.0001f;

                if (tan1ZeroCond && tan2ZeroCond)
                {
                    tan1 = Vector3.Normalize(pos1 - pos2) * small;
                    tan2 = Vector3.Normalize(pos2 - pos1) * small;
                } else
                {
                    if (tan1ZeroCond)
                        tan1 = Vector3.Normalize(pos1 - (pos2 - tan2 / 3.0f)) * small;

                    if (tan2ZeroCond)
                        tan2 = Vector3.Normalize(pos2 - (pos1 + tan1 / 3.0f)) * small;
                }

                if (tan1ZeroCond)
                {
                    data[off1 + 4] = tan1.X; data[off1 + 5] = tan1.Y; data[off1 + 6] = tan1.Z;
                }

                if (tan2ZeroCond)
                {
                    data[off2 + 4] = tan2.X; data[off2 + 5] = tan2.Y; data[off2 + 6] = tan2.Z;
                }
            }
        }

        public void FixZeroTangents()
        {
            for (var tubeId = 0; tubeId < tubeCount; ++tubeId)
                FixZeroTangents(tubeId);
        }

        public void SetCenter(Vector3 newCenter)
        {
            var center = Vector3.Zero;
            var totalSegmentCount = 0.0f;

            for (var tubeId = 0; tubeId < tubeCount; ++tubeId)
            {
                var baseNodeId = cummNodeCount[tubeId];
                var endNodeId = baseNodeId + nodeCount[tubeId] - 1;

                for (var nodeId = baseNodeId; nodeId < endNodeId; ++nodeId)
                {
                    var mapping = idToMemMap[nodeId];

                    var content = tubesBufferContent[mapping.BufferId];
                    var data = content.Content;


                    var off1 = mapping.Offset * 4;
                    var off2 = off1 + nodeElementCount * 4;

                    var pos1 = new Vector3(data[off1 + 0], data[off1 + 1], data[off1 + 2]);
                    var tan1 = new Vector3(data[off1 + 4], data[off1 + 5], data[off1 + 6]);

                    var pos2 = new Vector3(data[off2 + 0], data[off2 + 1], data[off2 + 2]);
                    var tan2 = new Vector3(data[off2 + 4], data[off2 + 5], data[off2 + 6]);

                    //center += (pos1 + pos2) * 0.5f;
                    center += (pos1 + pos2 + (pos1 + tan1 / 3.0f) + (pos2 - tan2 / 3.0f)) * 0.25f;

                    ++totalSegmentCount;
                }
            }


            center /= totalSegmentCount;

            var centerOff = newCenter - center;


            for (var nodeId = 0; nodeId < totalNodeCount; ++nodeId)
            {
                var mapping = idToMemMap[nodeId];

                var content = tubesBufferContent[mapping.BufferId];
                var data = content.Content;


                var off = mapping.Offset * 4;

                data[off + 0] += centerOff.X;
                data[off + 1] += centerOff.Y;
                data[off + 2] += centerOff.Z;

                if (mapping.Dublicated)
                {
                    content = tubesBufferContent[mapping.BufferId - 1];
                    data = content.Content;

                    off = (idToMemMap[nodeId - 1].Offset + nodeElementCount) * 4;

                    data[off + 0] += centerOff.X;
                    data[off + 1] += centerOff.Y;
                    data[off + 2] += centerOff.Z;
                }
            }
        }

        public void SetCenterToOrigin()
        {
            SetCenter(Vector3.Zero);
        }

        public void BindShaderAndShaderBuffer(Shader shader, int bufferIndex, string uniformBlockName)
        {
            shaderBuffer.BindToIndex(bufferIndex);

            shader.BindShaderBuffer(shaderBuffer, uniformBlockName);

            shader.Bind();
        }

#endif
    };

}
}
}
