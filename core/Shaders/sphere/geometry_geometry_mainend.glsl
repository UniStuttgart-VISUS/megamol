    //gl_Position = vec4((mins + maxs) * 0.5, 0.0, /*1.0*/inPos.w);
    //gl_PointSize = max((maxs.x - mins.x) * winHalf.x, (maxs.y - mins.y) * winHalf.y) * 0.5;
    
    // Cube vertices
    /*vec4 posA =  MVP * vec4(objPos.xyz + (camRight + camUp + camIn)*rad, 1.0);
    vec4 posB =  MVP * vec4(objPos.xyz + (camRight - camUp + camIn)*rad, 1.0);
    vec4 posC =  MVP * vec4(objPos.xyz + (-camRight + camUp + camIn)*rad, 1.0);
    vec4 posD =  MVP * vec4(objPos.xyz + (-camRight - camUp + camIn)*rad, 1.0);
    vec4 posE =  MVP * vec4(objPos.xyz + (-camRight - camUp - camIn)*rad, 1.0);
    vec4 posF =  MVP * vec4(objPos.xyz + (camRight - camUp - camIn)*rad, 1.0);
    vec4 posG =  MVP * vec4(objPos.xyz + (camRight + camUp - camIn)*rad, 1.0);
    vec4 posH =  MVP * vec4(objPos.xyz + (-camRight + camUp - camIn)*rad, 1.0);*/
    
    // Triangle strip
    /*gl_Position = posA; EmitVertex();
    gl_Position = posB; EmitVertex();
    gl_Position = posC; EmitVertex();
    gl_Position = posD; EmitVertex();
    gl_Position = posE; EmitVertex();
    gl_Position = posB; EmitVertex();
    gl_Position = posF; EmitVertex();
    gl_Position = posG; EmitVertex();
    gl_Position = posE; EmitVertex();
    gl_Position = posH; EmitVertex();
    gl_Position = posC; EmitVertex();
    gl_Position = posG; EmitVertex();
    gl_Position = posA; EmitVertex();
    gl_Position = posB; EmitVertex();
    gl_Position = posC; EmitVertex();
    gl_Position = posD; EmitVertex();
    gl_Position = posA; EmitVertex();
    gl_Position = posB; EmitVertex();*/
    
    // gl_Position = vec4(mins.x, maxs.y, 0.0, inPos.w); EmitVertex();
    // gl_Position = vec4(mins.x, mins.y, 0.0, inPos.w); EmitVertex();
    // gl_Position = vec4(maxs.x, maxs.y, 0.0, inPos.w); EmitVertex();
    // gl_Position = vec4(maxs.x, mins.y, 0.0, inPos.w); EmitVertex();

    //gl_Position = vec4((mins + maxs) * 0.5, projPos.z, (od > clipDat.w) ? 0.0 : 1.0);
    
    // Set gl_Position depending on flags (no fragment test required for visibility test)
    if (!(bool(flagsAvailable)) || (bool(flagsAvailable) && bitflag_isVisible(flag))) {
    
        gl_Position = vec4(mins.x, maxs.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
        gl_Position = vec4(mins.x, mins.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
        gl_Position = vec4(maxs.x, maxs.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
        gl_Position = vec4(maxs.x, mins.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
        EndPrimitive();
    }
}