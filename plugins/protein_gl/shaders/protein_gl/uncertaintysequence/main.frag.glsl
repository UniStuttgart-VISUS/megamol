#version 430

#extension GL_ARB_explicit_attrib_location : enable

#define STRUCT_COUNT 9                                     // must be equal to 'UncertaintyDataCall::secStructure::NOE'
#define EPS          0.0001

in vec4 quadPos;                                           // input from vertex shader

layout( location = 0 ) out vec4 fragColor;                 // output fragment color


uniform vec4[STRUCT_COUNT]  structCol;                     // the RGB colors for the strucutre types (order as in 'UncertaintyDataCall::secStructure')
uniform int[STRUCT_COUNT]   sortedStruct;                  // the sorted structure types from left to right corresponding to 'quadPos.x'
uniform float[STRUCT_COUNT] structUnc;                     // uncertainty values for structure types (order as in 'UncertaintyDataCall::secStructure')
uniform float               gradientInt;                   // gradient intervall in which color interpolation will be applied
uniform int                 colorInterpol;                 // option for choosing color interpolation method
                                                           // 0 -> UNCERTAIN_COLOR_RGB | 1 -> UNCERTAIN_COLOR_HSL | 2 -> UNCERTAIN_COLOR_HSL_HP

// function declaration
float hue2rgb(in float v1, in float v2, in float vH);
vec4  hsl2rgb(in vec4 hsla);
vec4  rgb2hsl(in vec4 rgba);
vec4  hpcb(in vec4 c1, in vec4 c2);                          // Hue Preserving Color Blending


void main()
{
    int structCount = STRUCT_COUNT;

    float fTemp     = quadPos.x;
    float start     = 0.0;
    float end       = 0.0;
    float middle    = 0.0;

    int diffStruct = 0;
    int index      = 0;

    vec4 startCol;
    vec4 endCol;
    vec4 tmpCol;

    float uTemp;
    float aStart;
    float aEnd;
    float xTemp;
    float mTemp;

    // determine structures with uncertainty > 0
    for (int k = 0; k < structCount; k++) {
        if (structUnc[sortedStruct[k]] > 0.0)
            diffStruct++;
    }

    index = 0;
    // loop through number of intervalls
    for (int k = 0; k < (diffStruct-1); k++) {

        // get data for first structure with uncertainty > 0
        while (structUnc[sortedStruct[index]] < EPS) {
            index++;
            if (index >= structCount) {
                break; // shouldn't happen?
            }
        }
        uTemp    = structUnc[sortedStruct[index]];
        start    = ((k == 0)?(0.0):(end));
        middle   = ((k == 0)?(uTemp):(start+(uTemp/2.0)));

        if(colorInterpol == 1) {
            startCol = rgb2hsl(structCol[sortedStruct[index]]);
        }
        else { // RGB
            startCol = structCol[sortedStruct[index]];
        }


        // get data for second structure with uncertainty > 0
        index++;
        while(structUnc[sortedStruct[index]] < EPS) {
            index++;
            if (index >= structCount) {
                break; // shouldn't happen?
            }
        }
        uTemp  = structUnc[sortedStruct[index]];
        end    = ((k == (diffStruct-2))?(middle+uTemp):(middle+(uTemp/2.0)));

        if(colorInterpol == 1) {
            endCol = rgb2hsl(structCol[sortedStruct[index]]);
        }
        else { // RGB
            endCol = structCol[sortedStruct[index]];
        }

        // check if fragment position in x direction is in current intervall
        if ((start <= fTemp) && (fTemp <= end)) {

            // calculate interpolation values
            xTemp  = (fTemp - start) / (end - start);      // is in [0,1]
            mTemp  = (middle - start) / (end - start);     // shift of x = 0 to 'middle'
            if (gradientInt > 0.0) {
                aStart = -((1.0 / gradientInt) * (xTemp - (mTemp + gradientInt/2.0)));
                aStart = (aStart > 1.0) ? (1.0) : ((aStart < 0.0) ? (0.0) : (aStart));
                aEnd   = 1.0 - aStart;                    // is already in [0,1]
            }
            else {
                aStart = (xTemp <= mTemp) ? (1.0) : (0.0);
                aEnd   = (xTemp > mTemp)  ? (1.0) : (0.0);
            }

            // interpolate colors depending on choice
            if (colorInterpol == 0) { // RGB linear
                tmpCol = (aStart*startCol) +  (aEnd*endCol);
            }
            else if (colorInterpol == 2) { // hue preserving
                tmpCol = hpcb((aStart*startCol), (aEnd*endCol));
            }
            else { // if (colorInterpol == 1) { // hsl linear

                tmpCol.y = aStart*startCol.y + aEnd*endCol.y; // interpolate S
                tmpCol.z = aStart*startCol.z + aEnd*endCol.z; // interpolate L

                // interpolate H (hue)
                if (startCol.y < EPS) { // startCol S = 0
                    tmpCol.x = endCol.x;
                }
                else if (endCol.y < EPS) { // endCol S = 0
                    tmpCol.x = startCol.x;
                }
                else if (startCol.x > endCol.x) {
                    //tmpCol.x = endCol.x + aStart*(startCol.x - endCol.x);
                    if ((startCol.x - endCol.x) < (360.0 - startCol.x + endCol.x)) {
                        tmpCol.x = endCol.x + aStart*(startCol.x - endCol.x);
                    }
                    else {
                        tmpCol.x = endCol.x - aStart*(360.0 - startCol.x + endCol.x);
                    }
                }
                else if (startCol.x < endCol.x){
                    //tmpCol.x = startCol.x + aEnd*(endCol.x - startCol.x);
                    if ((endCol.x - startCol.x) < (360.0 - endCol.x + startCol.x)) {
                        tmpCol.x = startCol.x + aEnd*(endCol.x - startCol.x);
                    }
                    else {
                        tmpCol.x = startCol.x - aEnd*(360.0 - endCol.x + startCol.x);
                    }
                }
                else { // startCol.x == endCol.x
                    tmpCol.x = startCol.x;
                }

                // adjust angle of hue to 0-360°
                tmpCol.x = (tmpCol.x < 0.0) ? (tmpCol.x + 360.0) : ((tmpCol.x >= 360.0) ? (tmpCol.x - 360.0) : (tmpCol.x));

                tmpCol = hsl2rgb(tmpCol);
            }
        fragColor = tmpCol;
        return;

        }
    }

    // fragColor = tmpCol;
    // fragColor = vec4(quadPos.x, quadPos.y, 0.0, 1.0);
}


// Hue Preserving Color Blending
vec4 hpcb(in vec4 c1, in vec4 c2) {

    vec4 cnew = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 HSL1 = rgb2hsl(c1);
    vec4 HSL2 = rgb2hsl(c2);

    float tmpHue;

    if (abs(HSL1.x - HSL2.x) < EPS) { // (HSL1.x == HSL2.x)
        cnew = c1 + c2;
    }
    else {
  if (HSL1.y >= HSL2.y) { // c1 is dominant
            tmpHue = (HSL1.x + 180.0);
            if (tmpHue >= 360.0) {
                tmpHue -= 360.0;
            }
            cnew = c1 + hsl2rgb(vec4(tmpHue, HSL2.y, HSL2.z, 1.0));
        }
        else if (HSL1.y < HSL2.y)  {// c2 is dominant
            tmpHue = (HSL2.x + 180.0);
            if (tmpHue >= 360.0) {
                tmpHue -= 360.0;
            }
            cnew = c2 + hsl2rgb(vec4(tmpHue, HSL1.y, HSL1.z, 1.0));
        }

    }

    return cnew;
}


// RGB to HSL
vec4 rgb2hsl(in vec4 rgba) {

    vec4 outCol = vec4(0.0, 0.0, 0.0, 1.0);
    float min, max, delta;
    float tmpR, tmpG, tmpB;

    min = (rgba.x < rgba.y) ? (rgba.x) : (rgba.y);
    min = (min < rgba.z )   ? (min)    : (rgba.z);

    max = (rgba.x > rgba.y) ? (rgba.x) : (rgba.y);
    max = (max > rgba.z)    ? (max)    : (rgba.z);
    delta = max - min;

    outCol.z = (max + min) / 2.0; // L

    if (delta < EPS)
    {
        outCol.x = 0.0; // H
        outCol.y = 0.0; // S
    }
    else
    {
        if (outCol.z < 0.5) { // L
            outCol.y = delta / (max + min); // S
        }
        else {
            outCol.y = delta / (2.0 - max - min); // L
        }

        tmpR = (((max - rgba.x) / 6.0) + (delta / 2.0)) / delta;
        tmpG = (((max - rgba.y) / 6.0) + (delta / 2.0)) / delta;
        tmpB = (((max - rgba.z) / 6.0) + (delta / 2.0)) / delta;

        if (rgba.x == max){
            outCol.x = tmpB - tmpG;
        }
        else if (rgba.y == max) {
            outCol.x = (1.0 / 3.0) + tmpR - tmpB;
        }
        else if (rgba.z == max) {
            outCol.x = (2.0 / 3.0) + tmpG - tmpR;
        }

        if (outCol.x < 0.0) {
            outCol.x += 1.0;
        }

        if (outCol.x >= 1.0) {
            outCol.x -= 1.0;
        }

        // if you want hue in range [0,1] comment line below
        outCol.x *= 360.0;
    }

    return outCol;
}


// HSL to RGB
vec4 hsl2rgb(in vec4 hsla) {

    float tmp1, tmp2;
    vec4 outCol = vec4(0.0, 0.0, 0.0, 1.0);

    if (hsla.y < EPS) { // S = 0
        outCol.x = hsla.z;   // L
        outCol.y = hsla.z;   // L
        outCol.z = hsla.z;   // L
    }
    else
    {
        if (hsla.z < 0.5) {
            tmp2 = hsla.z * (1.0 + hsla.y);
        }
        else {
            tmp2 = (hsla.z + hsla.y) - (hsla.y * hsla.z);
        }

        tmp1 = (2.0 * hsla.z) - tmp2;

        // if hue is in range [0,1] replace (hsla.x/360.0) with (hsla.x) in lines below
        outCol.x = hue2rgb(tmp1, tmp2, ((hsla.x/360.0) + (1.0 / 3.0)));
        outCol.y = hue2rgb(tmp1, tmp2, (hsla.x/360.0));
        outCol.z = hue2rgb(tmp1, tmp2, ((hsla.x/360.0) - (1.0 / 3.0)));
    }

    return outCol;
}


// HUE to RGB
float hue2rgb(in float v1, in float v2, in float vH) {

    float tmpH = vH;

    if (tmpH < 0.0) {
        tmpH += 1.0;
    }

    if (tmpH >= 1.0) {
        tmpH -= 1.0;
    }

    if ((6.0 * tmpH) < 1.0) {
        return (v1 + ((v2 - v1) * 6.0 * tmpH));
    }
    else if ((2.0 * tmpH) < 1.0) {
        return (v2);
    }
    else if ((3.0 * tmpH) < 2.0) {
        return (v1 + ((v2 - v1) * ((2.0 / 3.0) - tmpH) * 6.0));
    }
    else {
        return (v1);
    }
}
