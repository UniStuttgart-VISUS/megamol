vec3 decodeColor( float codedColor) {
    float col = codedColor;
    float red, green;
    if(col >= 1000000.0)
        red = floor(col / 1000000.0);
    else
        red = 0.0;
    col = col - (red * 1000000.0);
    if(col >= 1000.0)
        green = floor(col / 1000.0);
    else
        green = 0.0;
    col = col - (green * 1000.0);
    if(col > 256.0)
        col = 0.0;
    return vec3(red / 255.0, green / 255.0, col / 255.0);
}
