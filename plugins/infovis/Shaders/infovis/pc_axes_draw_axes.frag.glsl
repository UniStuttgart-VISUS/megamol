#version 430

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::earlyFragmentTests" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />

out vec4 fragColor;
in vec4 actualColor;

void main()
{
    fragColor = actualColor;
}
