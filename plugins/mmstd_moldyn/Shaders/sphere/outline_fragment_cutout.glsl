
    //float m_pi = 3.14159265358;
    //float wid = m_pi / 64;
    //float azimuth = atan(normal.y, normal.x);
    //azimuth = azimuth + m_pi; // scale up to range (0,2pi)
    //float inclination = acos(normal.z);
    //
    //bool throwaway = false;
    //float divisions = 8.0;
    //for (int i = 0; i < int(divisions + 0.5); i++) {
    //    float curazi = float(i) * (2.0 * m_pi / divisions);
    //    float nextazi = float(i + 1) * (2.0 * m_pi / divisions);
    //    if (azimuth > curazi + wid && azimuth < nextazi - wid) {
    //        throwaway = true;
    //    }
    //}
    //
    //float wf = 4.0f;
    //if (inclination < wf * wid || inclination > m_pi - wf * wid) throwaway = false;
    //if (inclination > (m_pi / 2.0) - wid && inclination < (m_pi / 2.0) + wid) throwaway = false;
    //
    //if (throwaway) discard;

    if (radicand > outlineSize) discard;
    // TODO make this independent of the sphere radius
