
void main(void) {

    // remove the sphere radius from the w coordinates to the rad varyings
    objPos = position;
    rad = (inGlobalRadius == 0.0) ? objPos.w : inGlobalRadius;
    objPos.w = 1.0;

	vertColor = color;
	if (inUseGlobalColor)
		vertColor = inGlobalColor;
	else
	if (inUseTransferFunction) {
		float texOffset = 0.5/float(textureSize(inTransferFunction, 0));
		float normPos = (color.r - inIndexRange.x)/(inIndexRange.y - inIndexRange.x);
		vertColor = texture(inTransferFunction, normPos * (1.0 - 2.0*texOffset) + texOffset);
	}

#ifdef WITH_SCALING
    rad *= scaling;
#endif // WITH_SCALING

    squarRad = rad * rad;

    // calculate cam position 
    camPos = MVinv[3]; // (C) by Christoph 
    camPos.xyz -= objPos.xyz; // cam pos to glyph space 
