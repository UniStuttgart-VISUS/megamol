#include "stdafx.h"
#include "IndexBuffer.h"

megamol::beztube::salm::IndexBuffer::IndexBuffer() : handle(0), count(0) {
    ::glGenBuffers(1, &handle);
}

megamol::beztube::salm::IndexBuffer::~IndexBuffer() {
    ::glDeleteBuffers(1, &handle);
}

void megamol::beztube::salm::IndexBuffer::CreateQuadCylinder(unsigned int vertexCountX, unsigned int vertexCountY) {
    unsigned int *indices = new unsigned int[vertexCountX * vertexCountY * 4];

    auto quadCountX = vertexCountX;
    auto quadCountY = vertexCountY - 1;

    auto quadOff = 0;

    for (auto y = 0u; y < quadCountY; ++y)
    {
        for (auto x = 0u; x < quadCountX; ++x)
        {
            auto i0 = x + y * vertexCountX;
            auto i1 = x + (y + 1) * vertexCountX;

            int i2, i3;
            if (x + 1 < quadCountX)
            {
                i2 = i1 + 1;
                i3 = i0 + 1;
            } else
            {
                i2 = (y + 1) * vertexCountX;
                i3 = y * vertexCountX;
            }

            indices[quadOff + 0] = static_cast<unsigned int>(i0);
            indices[quadOff + 1] = static_cast<unsigned int>(i1);
            indices[quadOff + 2] = static_cast<unsigned int>(i2);
            indices[quadOff + 3] = static_cast<unsigned int>(i3);

            quadOff += 4;
        }
    }

    SetBufferData(indices, vertexCountX * vertexCountY * 4);

    delete[] indices;
}

void megamol::beztube::salm::IndexBuffer::CreateClosedQuadCylinder(unsigned int vertexCountX, unsigned int vertexCountY) {
    unsigned int *indices = new unsigned int[(vertexCountX * vertexCountY + vertexCountX) * 4];

    auto capQuadCount = vertexCountX / 2;

    auto quadCountX = vertexCountX;
    auto quadCountY = vertexCountY - 1;


    auto quadOff = 0;

    unsigned int bottomCenterVertexId = (unsigned int)(vertexCountX * vertexCountY);

    for (auto i = 0u; i < capQuadCount; ++i)
    {
        unsigned int j = (unsigned int)i * 2u;

        indices[quadOff + 0] = bottomCenterVertexId;
        indices[quadOff + 1] = j;
        indices[quadOff + 2] = j + 1u;
        indices[quadOff + 3] = i == capQuadCount - 1 ? 0u : j + 2u;

        quadOff += 4;
    }


    for (auto y = 0u; y < quadCountY; ++y)
    {
        for (auto x = 0u; x < quadCountX; ++x)
        {
            auto i0 = x + y * vertexCountX;
            auto i1 = x + (y + 1) * vertexCountX;

            int i2, i3;
            if (x + 1 < quadCountX)
            {
                i2 = i1 + 1;
                i3 = i0 + 1;
            } else
            {
                i2 = (y + 1) * vertexCountX;
                i3 = y * vertexCountX;
            }

            indices[quadOff + 0] = (unsigned int)i0;
            indices[quadOff + 1] = (unsigned int)i1;
            indices[quadOff + 2] = (unsigned int)i2;
            indices[quadOff + 3] = (unsigned int)i3;

            quadOff += 4;
        }
    }


    unsigned int topCenterVertexId = bottomCenterVertexId + 1u;
    unsigned int topStartVertexId = bottomCenterVertexId - (unsigned int)vertexCountX;

    for (auto i = 0u; i < capQuadCount; ++i)
    {
        unsigned int j = topStartVertexId + (unsigned int)i * 2u;

        indices[quadOff + 0] = topCenterVertexId;
        indices[quadOff + 1] = i == capQuadCount - 1 ? topStartVertexId : j + 2u;
        indices[quadOff + 2] = j + 1u;
        indices[quadOff + 3] = j;

        quadOff += 4;
    }

    SetBufferData(indices, vertexCountX * vertexCountY * 4);

    delete[] indices;
}

void megamol::beztube::salm::IndexBuffer::Bind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);
}

void megamol::beztube::salm::IndexBuffer::Unbind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void megamol::beztube::salm::IndexBuffer::SetBufferData(unsigned int *data, unsigned int cnt) {
    Bind();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cnt * sizeof(unsigned int), data, GL_STATIC_DRAW);
    count = cnt;
    Unbind();
}
