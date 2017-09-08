#pragma once

#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace beztube {
namespace salm {

    class IndexBuffer {
    public:
        IndexBuffer();
        ~IndexBuffer();

        void CreateQuadCylinder(unsigned int vertexCountX, unsigned int vertexCountY);
        void CreateClosedQuadCylinder(unsigned int vertexCountX, unsigned int vertexCountY);

        void Bind();
        void Unbind();
        void SetBufferData(unsigned int *data, unsigned int cnt);

        inline GLuint Handle() const { return handle; }
        inline unsigned int Count() const { return count; }

    private:
        GLuint handle;
        unsigned int count;
    };

}
}
}
