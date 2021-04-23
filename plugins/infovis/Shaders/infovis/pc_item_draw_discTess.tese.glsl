#version 450

#include <snippet name="::core_utils::tflookup" />
#include <snippet name="::core_utils::tfconvenience" />
#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::common" />
#include <snippet name="::pc_item_draw::tessuniforms" />
#include <snippet name="::bitflags::main" />

// BEGIN Output data
layout(isolines, equal_spacing) in;
//patch in InterfaceTEI
//{
//  flat uint baseItemID;
//  flat uint generatedLines;
//} in_;
struct InterfaceTEI
{
    uint baseItemID;
    uint generatedLines;
};
layout(location = 0) patch in InterfaceTEI in_;

struct InterfaceTEO
{
    uint itemID;
    vec4 color;
};
layout(location = 0) flat out InterfaceTEO out_;
// END Output data

void main(void) {
    uint itemID = int(round(in_.baseItemID + gl_TessCoord.y * in_.generatedLines));
    uint worldSpaceAxis = uint(gl_TessCoord.x * 5); //int(round(gl_TessCoord.x * dimensionCount));
    uint dataDimension = pc_dimension(worldSpaceAxis);
    uint dataID = pc_item_dataID(itemID, dataDimension);
    out_.itemID = itemID;

    float value = itemID / float(itemCount);
    out_.color = tflookup(value);
    out_.color = vec4(1.0, 0.0, 0.0, 1.0);

    vec4 vertex = pc_item_vertex(itemID, dataID, dataDimension, worldSpaceAxis);
    //vertex = vec4(
    //  margin.x + axisDistance * gl_TessCoord.x * 100
    //  , margin.y + axisHeight * gl_TessCoord.y * 100
    //  , pc_item_defaultDepth
    //  , 1.0
    //);
    //out_.color = vec4(1.0, 0.0, 0.0, 1.0);

    gl_Position = projection * modelView * vertex;
}
