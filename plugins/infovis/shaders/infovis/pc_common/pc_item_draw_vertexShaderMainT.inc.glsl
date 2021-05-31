uniform vec2 res = vec2(1,1);
uniform float widthR = 1.0;
uniform float heightR = 1.0;
uniform float thicknessP;

void main() {
#define DOLINES
//#define DOSTRIPS

    // id of line segment
    uint instanceID = gl_VertexID / ((dimensionCount - 1) * 6);
    out_.itemID = instanceID;
    // ID of Dimension being look at
    uint theID = gl_VertexID % ((dimensionCount - 1) * 6);
    theID = (theID / 6) + ((theID % 6) / 2) - ((theID % 6) / 3);

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

    // aktuell betrachteter Vertex
    vec4 vertex = pc_item_vertex(out_.itemID, out_.dataID, out_.dataDimension, out_.worldSpaceAxis);
    // Vertices auf der linken und rechten Achse
    vec4 vertexL = pc_item_vertex(out_.itemID, pc_item_dataID(out_.itemID, pc_dimension(theID-1)), pc_dimension(theID-1), out_.worldSpaceAxis-1);
    vec4 vertexR = pc_item_vertex(out_.itemID, pc_item_dataID(out_.itemID, pc_dimension(theID+1)), pc_dimension(theID+1), out_.worldSpaceAxis+1);

    //vertex.y += (gl_VertexID % 2) * axisHeight * 0.002;

    vec2 tempL = vec2(0, 0.0);
    vec2 tempR = vec2(0.0, 0.0);
    // magic number, to be changed to resolution based calculation
    float thickness = thicknessP * sqrt(1/pow(widthR,2) + 1/pow(heightR,2));

    if (out_.worldSpaceAxis != 0 && out_.worldSpaceAxis != dimensionCount) {
        // Vektoren von linken zum aktuellen Vertex, dies ist die Linie die von links kommt
        tempL =  (projection * modelView * vertex - projection * modelView * vertexL).xy;
        // Linie zur naechsten Achse
        tempR =  (projection * modelView * vertexR - projection * modelView * vertex).xy;

        tempL = normalize(tempL);
        tempR = normalize(tempR);
        tempL.x *= widthR;
        tempR.x *= widthR;
        tempL.y *= heightR;
        tempR.y *= heightR;

        // Bestimmung der Orthogonalen durch Drehung
        //vec2 oL = vec2(-tempL.y * res.y / res.x, tempL.x * res.x / res.y);
        vec2 oL = vec2(-tempL.y, tempL.x);
        //vec2 oR = vec2(-tempR.y * res.y/res.x, tempR.x*res.x/res.y);
        vec2 oR = vec2(-tempR.y, tempR.x);

        // benoetigt zur berechnung des laenge des offset vektors
        oL = thickness * normalize(oL);
        oR = thickness * normalize(oR);

        // h ist winkelhalbierende
        vec2 h = normalize(oL + oR);
        //out_.color = vec4(heightR / widthR, 0.0 , 0.0, 1.0);

        // laenge der winkelhalbierenden um dicke zu garantieren, berechnet aus Winkel der rechten orthogonalen und der winkelhalbierenden
        h = h * thickness / dot(oR, h) * (length(oR) * length(h));
        h.x *= 1.0 * heightR / widthR ;
        tempL = h;
    }

    if (out_.worldSpaceAxis == 0) {
        tempR = (projection * modelView * vertexR - projection * modelView * vertex).xy;
        tempR = normalize(tempR);
        tempR.x *= widthR;
        tempR.y *= heightR;
        vec2 oR = vec2(-tempR.y, tempR.x);
        oR = thickness * normalize(oR);
        tempL = vec2(0.0, 1.0) * thickness * thickness / dot(oR, vec2(0.0, 1.0));
        //out_.color = vec4(pc_item_dataValue(pc_item_dataID(out_.itemID, pc_dimension(theID + 1)), pc_dimension(theID + 1)), 0.0, 1.0, 1.0);
    }
    if (out_.worldSpaceAxis == dimensionCount - 1) {
        //tempL =  (projection * modelView * vertex - projection * modelView * vertexL).xy;
        //vec2 oL = vec2(-tempL.y, tempL.x);
        //tempL = vec2(0.0, 1.0) * thickness * thickness / dot(oL, vec2(0.0, 1.0));
        //out_.color = vec4(1.0, 0.0, 1.0, 1.0);
        tempL = (projection * modelView * vertex - projection * modelView * vertexL).xy;
        tempL = normalize(tempL);
        tempL.x *= widthR;
        tempL.y *= heightR;
        vec2 oL = vec2(-tempL.y, tempL.x);
        oL = thickness * normalize(oL);
        tempL = vec2(0.0, 1.0) * thickness * thickness / dot(oL, vec2(0.0, 1.0));
    }
    vec2 offset = (-0.5 + (gl_VertexID % 2)) * tempL;
    //offset.y *= 2.3839096683;
    //vertex = vertex + vec4(offset, 0.0, 0.0);

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
    //vertex = vertexL;
    gl_Position = projection * modelView * (vertex) + (vec4(offset, 0.0, 0.0));
}
