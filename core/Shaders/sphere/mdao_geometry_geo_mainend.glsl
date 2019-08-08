    // clipping
    float od = clipDat.w - 1.0;
    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        od = dot(objPos.xyz, clipDat.xyz) - rad;
    }
	
    //gl_Position = vec4((mins + maxs) * 0.5, projPos.z, (od > clipDat.w) ? 0.0 : 1.0);
    gl_Position = vec4(mins.x, maxs.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
    gl_Position = vec4(mins.x, mins.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
    gl_Position = vec4(maxs.x, maxs.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
    gl_Position = vec4(maxs.x, mins.y, projPos.z, (od > clipDat.w) ? 0.0 : 1.0); EmitVertex();
    EndPrimitive();
}		