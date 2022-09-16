uniform sampler3D potentialTex0;
uniform sampler3D potentialTex1;
uniform int colorMode;
uniform int renderMode;
uniform int unmappedTrisColorMode;
uniform vec3 colorMin;
uniform vec3 colorMax;
uniform vec3 colorUniform;
uniform float minPotential;
uniform float maxPotential;
uniform float alphaScl;
uniform float maxPosDiff;
uniform int uncertaintyMeasurement;

varying vec3 lightDir;
varying vec3 view;
varying vec3 normalFrag;
varying vec3 posNewFrag; // Interpolated WS position
varying float pathLenFrag;
varying float surfAttribFrag;
varying float corruptFrag;

void main() {

    vec4 lightparams, color;

    if (renderMode == 1) { // Points
        lightparams = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (renderMode == 2) { // Wireframe
        lightparams = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (renderMode == 3) { // Surface
        lightparams = vec4(0.2, 0.8, 0.0, 10.0);
    }

    // Determine color
    if (colorMode == 0) { // Uniform color

        //color = vec4(colorUniform, 1.0 - pathLenFrag/maxPosDiff);
        color = vec4(colorUniform, 1.0);

    } else if (colorMode == 1) { // Normal
        //lightparams = vec4(1.0, 0.0, 0.0, 1.0);
        color = vec4(normalize(normalFrag), 1.0);
        color = vec4(colorUniform, 1.0 - pathLenFrag/maxPosDiff); // DEBUG
    } else if (colorMode == 2) { // Texture coordinates
        lightparams = vec4(1.0, 0.0, 0.0, 1.0);
        color = vec4(gl_TexCoord[0].stp, 1.0);
    } else if (colorMode == 4) { // Uncertainty
        float diff;
  //      if (uncertaintyMeasurement == 0) { // Euclidean distance
  //          diff = length(posOldFrag-posNewFrag);
  //      } else if (uncertaintyMeasurement == 1) { // Path length
            diff = pathLenFrag;
  //      }
        color = MixColors(diff, 0.0, maxPosDiff*0.5,
            maxPosDiff,
            vec4(1.0, 1.0, 1.0, 1.0),
            vec4(1.0, 1.0, 0.0, 1.0),
            vec4(1.0, 0.0, 0.0, 1.0));

        float potDiff = diff;

        //vec3 colOrangeMsh = vec3(100, 0.9746*potDiff/(maxPotential-minPotential), 0.8968);
        vec3 colYellowMsh = vec3(102.44, 0.6965*potDiff/maxPosDiff, 1.5393);
        //vec3 blueMsh = vec3(90, 1.08*potDiff/maxPosDiff, -1.1);

        color = vec4(MSH2RGB(colYellowMsh.r, colYellowMsh.g, colYellowMsh.b), 1.0);

    } else if (colorMode == 5) { // Surface potential 0

        // Interpolation in MSH color space
        vec3 colMsh = CoolWarmMsh(texture3D(potentialTex0, gl_TexCoord[0].stp).a,
                        minPotential, 0.0, maxPotential);
        color = vec4(MSH2RGB(colMsh.x, colMsh.y, colMsh.z), 1.0);

    } else if (colorMode == 6) { // Surface potential 1

        // Interpolation in MSH color space
        vec3 colMsh = CoolWarmMsh(surfAttribFrag,
              minPotential, 0.0, maxPotential);
        color = vec4(MSH2RGB(colMsh.x, colMsh.y, colMsh.z), 1.0);

    } else if (colorMode == 7) { // Surface potential difference

        float potDiff = surfAttribFrag;

        //vec3 colOrangeMsh = vec3(100, 0.9746*potDiff/(maxPotential-minPotential), 0.8968);
        vec3 colYellowMsh = vec3(102.44, 0.6965*potDiff/(maxPotential-minPotential), 1.5393);
        //vec3 blueMsh = vec3(90, 1.08*potDiff/(maxPotential-minPotential), -1.1);

        color = vec4(MSH2RGB(colYellowMsh.r, colYellowMsh.g, colYellowMsh.b), 1.0 - pathLenFrag/maxPosDiff);

    } else if (colorMode == 8) { // Surface potential sign switch

        //lightparams = vec4(0.7, 0.4, 0.0, 1.0);
        //float potOld = texture3D(potentialTex1, gl_TexCoord[1].stp).a;
        //float potNew = texture3D(potentialTex0, gl_TexCoord[0].stp).a;
        //float potDiff = abs(potOld-potNew);
        // Calc euclidian distance
        // Calc euclidian distance
        float posDiff;
//        if (uncertaintyMeasurement == 0) { // Euclidean distance
//            posDiff = length(posOldFrag-posNewFrag);
//        } else if (uncertaintyMeasurement == 1) { // Path length
            posDiff = pathLenFrag;
//        }

        float signSwitchedFlag = surfAttribFrag;


        //vec3 colTurquoiseMsh = vec3(109.81, 0.9746*signSwitchedFlag, 0.8968);
        //vec3 colDiffMsh = (1.0 - posDiff/maxPosDiff)*colOrangeMsh + (posDiff/maxPosDiff)*colWhiteMsh;

        // Green
        //vec3 colorSign = vec3(0.57, 0.76, 0.0)*signSwitchedFlag + (1.0-signSwitchedFlag)*vec3(1.0, 1.0, 1.0);

        // Yellow
        //vec3 colorSign = vec3(1.0, 0.84, 0.0)*signSwitchedFlag + (1.0-signSwitchedFlag)*vec3(1.0, 1.0, 1.0);
        vec3 colorSign = vec3(0.96, 0.74, 0.06)*signSwitchedFlag + (1.0-signSwitchedFlag)*vec3(1.0, 1.0, 1.0);

        color = vec4(colorSign, 1.0 - posDiff/maxPosDiff);

        // Draw interior darker
        //if (dot(view, normalFrag) > 0) {
        //    colDiff *= 0.75;
        //}

        //color = vec4(colDiff, 1.0 - posDiff/maxPosDiff);
        //color = vec4(colDiff, 1.0);

        // Set for corrupt triangles
        //color = color*(1.0 - corruptTriangleFlagFrag) +
        //        vec4(1.0, 0.0, 1.0, 1.0)*corruptTriangleFlagFrag;
        //color.a *= (1.0 - corruptTriangleFlagFrag);
        //color.rgb *= 1.0 - corruptTriangleFlagFrag;
    } else if (colorMode == 9) { // Mesh laplacian
        float potDiff = surfAttribFrag;

        //vec3 colOrangeMsh = vec3(100, 0.9746*(1.0-potDiff/(maxPotential-minPotential)), 0.8968);
        vec3 colYellowMsh = vec3(102.44, 0.6965*(potDiff/(maxPotential-minPotential)), 1.5393);
        //vec3 blueMsh = vec3(90, 1.08*potDiff/(maxPotential-minPotential), -1.1);

        color = vec4(MSH2RGB(colYellowMsh.r, colYellowMsh.g, colYellowMsh.b), 1.0);
    } else { // Invalid color mode
        color = vec4(0.5, 1.0, 1.0, 1.0);
    }

   //color.a *= (1.0 - corruptFrag);
   // Alpha scaling by the user
    color.a *= alphaScl;
    vec3 n = normalFrag;

    vec4 corruptColor;
    if (unmappedTrisColorMode == 0) {
        corruptColor = color;
    } else if (unmappedTrisColorMode == 1) {
        corruptColor = vec4(color.rgb, 0.0);
    } else if (unmappedTrisColorMode == 2) {
        corruptColor = vec4(0.0, 0.458823529, 0.650980392, 1.0);
    }
    color = corruptFrag*corruptColor + (1.0-corruptFrag)*color;
    //color = vec4(0.0, 0.458823529, 0.650980392, 1.0);

    /*if (corruptFrag > 0) {
        // 230,97,1
        //color.rgb = vec3(0.90, 0.38, 0.0);

        //#e7be40
        // 231, 190, 64
        //color.rgb = vec3(0.905882353, 0.745098039, 0.250980392);
//        color.rgb = vec3(0.654901961, 0.494117647, 0.0);
        //color.rgb = vec3(0.0, 0.458823529, 0.650980392);
//        color.rgb = vec3(0.0, 0.5, 0.7);
    }*/

    if (gl_FrontFacing) {
        gl_FragColor = vec4(LocalLighting(normalize(view), normalize(n),
            normalize(lightDir), color.rgb, lightparams), color.a);
    } else {
        gl_FragColor = vec4(LocalLighting(normalize(view), normalize(-n),
            normalize(lightDir), color.rgb*0.7, lightparams), color.a);
    }






}
