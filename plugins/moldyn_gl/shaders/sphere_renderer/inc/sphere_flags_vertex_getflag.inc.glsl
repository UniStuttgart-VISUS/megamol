uint flag = uint(0);

#ifdef FLAGS_AVAILABLE
    if (bool(flags_enabled)) {
        flag = inFlags[(flags_offset + uint(gl_VertexID))];
    }
#endif // FLAGS_AVAILABLE
