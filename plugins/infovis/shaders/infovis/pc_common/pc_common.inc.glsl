uint pc_dimension_count()
{
    return dimensionCount;
}

uint pc_dimension(uint base)
{
    return axisIndirection[base];
}

uint pc_item_dataID(uint itemID, uint dimension)
{
    return itemID * dimensionCount + dimension;
}

float pc_item_dataValue(uint dataID, uint dimension)
{
    return (data[dataID] - dataMinimum[dimension]) / (dataMaximum[dimension] - dataMinimum[dimension]);
}

float pc_item_dataValue_unscaled(uint dataID) {
    return data[dataID];
}

vec4 pc_item_vertex(uint itemID, uint dataID, uint dimension, uint worldSpaceAxis)
{
    return vec4(
    margin.x + axisDistance * worldSpaceAxis
    , margin.y + axisHeight * pc_item_dataValue(dataID, dimension)
    , pc_item_defaultDepth
    , 1.0
    );
}
vec2 pc_item_vec2(uint itemID, uint dataID, uint dimension)
{
    //return vec2(
    //  abscissae[dimension]
    //  , pc_item_dataValue(dataID, dimension)
    //  );
    return vec2(1.0);
}

vec4 axis_line(uint dimension, int topFactor) {

    return vec4(
    margin.x + axisDistance * dimension,
    margin.y + topFactor * axisHeight,
    pc_item_defaultDepth,
    1.0
    );
}
