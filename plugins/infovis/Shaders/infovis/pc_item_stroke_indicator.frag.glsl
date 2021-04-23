#version 430

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::useLineStrip" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::common" />
#include <snippet name="::pc_item_stroke::uniforms" />

uniform vec4 indicatorColor = vec4(0.0, 0.0, 1.0, 1.0);

out vec4 fragColor;

void main()
{
    fragColor = indicatorColor;
}
