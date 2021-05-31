void main() {
#define DOLINES
//#define DOSTRIPS

#ifdef DOSTRIPS
    uint instanceID = gl_InstanceID; //getInstanceID();
    out_.itemID = instanceID;

#ifdef DOLINES // line strip
    uint theID = gl_VertexID;
#else          // triangle strip
    uint theID = gl_VertexID / 2;
#endif

#else // no strips
#ifdef DOLINES // single lines
    uint instanceID = gl_VertexID / ((dimensionCount - 1) * 2);
    out_.itemID = instanceID;
    uint theID = gl_VertexID % ((dimensionCount - 1) * 2);
    theID = (theID + 1) / 2;
#else          // single triangles
    uint instanceID = gl_VertexID / ((dimensionCount - 1) * 6);
    out_.itemID = instanceID;
    uint theID = gl_VertexID % ((dimensionCount - 1) * 6);
    theID = (theID / 6) + ((theID % 6) / 2) - ((theID % 6) / 3);
#endif
#endif

    out_.dataDimension = pc_dimension(theID);
    out_.worldSpaceAxis = theID;

    out_.dataID = pc_item_dataID(out_.itemID, out_.dataDimension);

    float value = out_.itemID / float(itemCount - 1);
    if (colorColumn != -1) {
        value = pc_item_dataValue_unscaled(pc_item_dataID(out_.itemID, colorColumn));
    }
    if (bitflag_test(flags[out_.itemID], FLAG_SELECTED, FLAG_SELECTED)) {
        out_.color = color;
    } else {
        out_.color = mix(color, tflookup(value), tfColorFactor);
    }

    vec4 vertex = pc_item_vertex(out_.itemID, out_.dataID, out_.dataDimension, out_.worldSpaceAxis);
#ifndef DOLINES // triangles need to be "less degenerate"
    vertex.y += (gl_VertexID % 2) * axisHeight * 0.002;
#endif

#if 0
    else
    {
        const vec4 center = vec4(vertex.x, 0.5 * scaling.y, vertex.z, vertex.w);
        float alpha = smoothstep(0.0, 1.0, float(tick) * 0.001f);
        vertex = mix(center, vertex, alpha);
    }
#endif
    if (bitflag_test(flags[out_.itemID], fragmentTestMask, fragmentPassMask)) {
        gl_ClipDistance[0] = 1.0;
    } else {
        gl_ClipDistance[0] = -1.0;
    }
    gl_Position = projection * modelView * vertex;
}
