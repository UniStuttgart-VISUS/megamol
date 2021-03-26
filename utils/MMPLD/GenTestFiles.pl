use strict;
use warnings qw(FATAL);
use Cwd;
use MMPLD;

my $outfile;
my $m;
my $numPoints = 4;

my @outfiles = ();

#add three lists of global-color particles along three edges of the box
#to check for proper setting of list-local global color :D

sub AddParticles {
    my $scale = shift;
    $m->AddParticle({
        x=>0.0, y=> 0.0, z=> 0.0, r=>1.0 * $scale, g=>1.0 * $scale, b=>1.0 * $scale, a=>1.0 * $scale, rad=>0.5, i=>255.0
    });
    $m->AddParticle({
        x=>1.0, y=> 0.0, z=> 0.0, r=>1.0 * $scale, g=>0.0 * $scale, b=>0.0 * $scale, a=>1.0 * $scale, rad=>0.2, i=>64.0
    });
    $m->AddParticle({
        x=>0.0, y=> 1.0, z=> 0.0, r=>0.0 * $scale, g=>1.0 * $scale, b=>0.0 * $scale, a=>1.0 * $scale, rad=>0.3, i=>128.0
    });
    $m->AddParticle({
        x=>0.0, y=> 0.0, z=> 1.0, r=>0.0 * $scale, g=>0.0 * $scale, b=>1.0 * $scale, a=>1.0 * $scale, rad=>0.4, i=>192.0
    });
}

sub AddEdgeLists {

    $m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>5, globalcolor=>"255 0 0 255"
            });
    for (my $x = 2.0; $x >= 1.2; $x -= 0.2) {
        $m->AddParticle({
            x=>$x, y=> -2.0, z=> -2.0
        });
    }

    $m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>5, globalcolor=>"0 255 0 255"
            });
    for (my $x = 2.0; $x >= 1.2; $x -= 0.2) {
        $m->AddParticle({
            x=>-2.0, y=>$x, z=> -2.0
        });
    }

    $m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>5, globalcolor=>"0 0 255 255"
            });
    for (my $x = 2.0; $x >= 1.2; $x -= 0.2) {
        $m->AddParticle({
            x=>-2.0, y=> -2.0, z=>$x
        });
    }
}

my $numLists = 4;

$outfile = "test_xyz_float_rgb_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_RGB_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_double_rgb_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_DOUBLE, colortype=>$COLOR_RGB_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_rgb_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_RGB_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_rgba_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_RGBA_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_double_rgba_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_DOUBLE, colortype=>$COLOR_RGBA_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_rgba_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_RGBA_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_rgba_byte.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_RGBA_BYTE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(255.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_double_rgba_byte.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_DOUBLE, colortype=>$COLOR_RGBA_BYTE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(255.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_rgba_byte.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_RGBA_BYTE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(255.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_int_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_double_int_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_DOUBLE, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_double_int_double.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_DOUBLE, colortype=>$COLOR_INTENSITY_DOUBLE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_int_float.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_none.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints, globalcolor=>"255 255 0 255"
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_double_rgba_ushort.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_DOUBLE, colortype=>$COLOR_RGBA_USHORT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(65535.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_double_none.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZ_DOUBLE, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints, globalcolor=>"255 255 0 255"
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_none.mmpld";
print "writing $outfile\n";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>$numLists});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints, globalcolor=>"255 255 0 255"
            });
AddParticles(1.0);
AddEdgeLists();
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

open my $batch, ">", "SphereTest.bat" or die "cannot open batch file";

my @renderer_modes = ("SimpleSphere", "GeometryShaderSphere", "SSBOSphere", "SSBOSphereStatic","BufferArraySphere", "AmbientOcclusionSphere", "OSPRayGeometrySphere", "OSPRayNHGeometrySphere");
foreach my $r (@renderer_modes) {
    foreach my $f (@outfiles) {
        my $proj = "$f-$r.lua";
        open my $fh, ">", $proj or die "cannot open $proj";
        
        print $fh qq{mmCreateView("test", "View3DGL", "::view")\n};
        print $fh qq{mmCreateModule("MMPLDDataSource", "::dat")\n};

        if ($r =~ /^OSPRay/) {

            print $fh qq{mmCreateModule("BoundingBoxRenderer","::bbox")\n};
            print $fh qq{mmCreateModule("ContextToGL","::c2gl")\n};
            print $fh qq{mmCreateModule("OSPRayRenderer", "::osp")\n};
            if ($r =~ /^OSPRayGeometrySphere/) {
                print $fh qq{mmCreateModule("OSPRaySphereGeometry", "::renderer")\n};
            }
            elsif ($r =~ /^OSPRayNHGeometrySphere/) {
                print $fh qq{mmCreateModule("OSPRayNHSphereGeometry", "::renderer")\n};
            }

            print $fh qq{mmCreateModule("AmbientLight", "::amb")\n};
            print $fh qq{mmCreateModule("OSPRayOBJMaterial", "::mat")\n};

            print $fh qq{mmCreateCall("CallRender3DGL","::view::rendering","::bbox::rendering")\n};
            print $fh qq{mmCreateCall("CallRender3DGL","::bbox::chainRendering","::c2gl::rendering")\n};
            print $fh qq{mmCreateCall("CallRender3D","::c2gl::getContext","::osp::rendering")\n};
            print $fh qq{mmCreateCall("CallOSPRayStructure", "::osp::getStructure", "::renderer::deployStructureSlot")\n};
            print $fh qq{mmCreateCall("CallLight", "::osp::lights", "::amb::deployLightSlot")\n};
            print $fh qq{mmCreateCall("CallOSPRayMaterial", "::renderer::getMaterialSlot", "::mat::deployMaterialSlot")\n};

            print $fh qq{mmSetParamValue("::osp::useDBcomponent", "false")\n};

        } else {

            print $fh qq{mmCreateModule("BoundingBoxRenderer","::bbox")\n};
            print $fh qq{mmCreateModule("DistantLight","::distlight")\n};
            print $fh qq{mmCreateModule("SphereRenderer", "::renderer")\n};

            print $fh qq{mmCreateCall("CallRender3DGL", "::view::rendering", "::bbox::rendering")\n};
            print $fh qq{mmCreateCall("CallRender3DGL","::bbox::chainRendering","::renderer::rendering")\n};
            print $fh qq{mmCreateCall("CallLight","::renderer::lights","::distlight::deployLightSlot")\n};   

            if ($r =~ /^SimpleSphere/) {
                print $fh qq{mmSetParamValue("::renderer::renderMode", "Simple")\n};
            }
            elsif ($r =~ /^GeometryShaderSphere/) {
                print $fh qq{mmSetParamValue("::renderer::renderMode", "Geometry_Shader")\n};
            }
            elsif ($r =~ /^SSBOSphere/) {
                print $fh qq{mmSetParamValue("::renderer::renderMode", "SSBO_Stream")\n};
                if ($r =~ /^SSBOSphereStatic/) {
                    print $fh qq{mmSetParamValue("::renderer::ssbo::staticData", "true")\n};
                }
            }
            elsif ($r =~ /^BufferArraySphere/) {
                print $fh qq{mmSetParamValue("::renderer::renderMode", "Buffer_Array")\n};
            }
            elsif ($r =~ /^AmbientOcclusionSphere/) {
                print $fh qq{mmSetParamValue("::renderer::renderMode", "Ambient_Occlusion")\n};
                print $fh qq{mmSetParamValue("::renderer::ambient occlusion::enableLighting", "true")\n};
            }      
               
            print $fh qq{mmSetParamValue("::distlight::Direction", "-0.500000;0.500000;0.000000")\n};                   
        }
        print $fh qq{mmCreateCall("MultiParticleDataCall", "::renderer::getdata", "::dat::getData")\n};   

        print $fh qq{mmSetParamValue("::view::camstore::settings", [=[{"centre_offset":[0.0,0.0],"convergence_plane":0.0,"eye":0,"far_clipping_plane":12.97671890258789,"film_gate":[0.0,0.0],"gate_scaling":0,"half_aperture_angle":0.2617993950843811,"half_disparity":0.02500000037252903,"image_tile":[0,720,1280,0],"near_clipping_plane":0.012976719066500664,"orientation":[0.17020022869110107,-0.24738462269306183,-0.0711577907204628,-0.9511932134628296],"position":[4.224766731262207,3.3975491523742676,7.757389545440674],"projection_type":0,"resolution_gate":[1280,720]}]=])\n};
        print $fh qq{mmSetParamValue("::view::camstore::autoLoadSettings", "true")\n};
        print $fh qq{mmSetParamValue("::dat::filename", "}.getcwd().qq{/$f")\n};

        print $fh qq{mmShowGUI(false)\n};                          # print $fh qq{mmSetGUIVisible(false)\n};
        print $fh qq{mmRenderNextFrame()\n};
        print $fh qq{mmScreenShot("}.getcwd().qq{/$f-$r.png")\n};  # print $fh qq{mmScreenshot("}.getcwd().qq{/$f-$r.png")\n};
        print $fh qq{mmQuit()\n};

        close $fh;

        print $batch qq{megamol.exe ..\\..\\..\\utils\\MMPLD\\$proj\n};
    }
}