use strict;
use warnings qw(FATAL);
use Cwd;
use MMPLD;
use File::Basename;
use File::Spec;

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

sub MakeTest {
    my %params = %{$_[0]};
    $outfile = "../../test/data/test_" . $VERTEX_FORMAT_NAMES[$params{vt}] . "_" . $COLOR_FORMAT_NAMES[$params{ct}] . ".mmpld";
    $outfile = File::Spec->rel2abs($outfile);
    print "writing $outfile\n";
    push @outfiles, $outfile;
    $m = MMPLD->new({filename=>$outfile, numframes=>1});
    $m->StartFrame({frametime=>1.23, numlists=>$numLists});

    if ($params{ct} == $COLOR_NONE) {
        $m->StartList({
                    vertextype=>$params{vt}, colortype=>$params{ct}, globalradius=>$params{grad},
                    minintensity=>$params{minint}, maxintensity=>$params{maxint}, particlecount=>$numPoints,
                    globalcolor=>"255 255 0 255"
                    });
    } else {
        $m->StartList({
                    vertextype=>$params{vt}, colortype=>$params{ct}, globalradius=>$params{grad},
                    minintensity=>$params{minint}, maxintensity=>$params{maxint}, particlecount=>$numPoints
                    });
    }
    AddParticles($params{colscale});
    AddEdgeLists();
    $m->OverrideBBox(-2,-2,-2,2,2,2);
    $m->Close();
}

my @tests = (
    {vt=>$VERTEX_XYZ_FLOAT, ct=>$COLOR_RGB_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_DOUBLE, ct=>$COLOR_RGB_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZR_FLOAT, ct=>$COLOR_RGB_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_FLOAT, ct=>$COLOR_RGBA_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_DOUBLE, ct=>$COLOR_RGBA_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZR_FLOAT, ct=>$COLOR_RGBA_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_FLOAT, ct=>$COLOR_RGBA_BYTE, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>255.0},
    {vt=>$VERTEX_XYZ_DOUBLE, ct=>$COLOR_RGBA_BYTE, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>255.0},
    {vt=>$VERTEX_XYZR_FLOAT, ct=>$COLOR_RGBA_BYTE, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>255.0},
    {vt=>$VERTEX_XYZ_FLOAT, ct=>$COLOR_INTENSITY_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_DOUBLE, ct=>$COLOR_INTENSITY_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_DOUBLE, ct=>$COLOR_INTENSITY_DOUBLE, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZR_FLOAT, ct=>$COLOR_INTENSITY_FLOAT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_FLOAT, ct=>$COLOR_NONE, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZ_DOUBLE, ct=>$COLOR_RGBA_USHORT, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>65535.0},
    {vt=>$VERTEX_XYZ_DOUBLE, ct=>$COLOR_NONE, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
    {vt=>$VERTEX_XYZR_FLOAT, ct=>$COLOR_NONE, grad=>0.1, minint=>0.0, maxint=>255.0, colscale=>1.0},
);

foreach my $t (@tests) {
    MakeTest($t);
}

open my $batch, ">", "SphereTest.bat" or die "cannot open batch file";

my @renderer_modes = ("SimpleSphere", "GeometryShaderSphere", "SSBOSphere", "SSBOSphereStatic","BufferArraySphere", "AmbientOcclusionSphere", "OSPRayGeometrySphere", "OSPRayNHGeometrySphere", "Splats", "Outline");
foreach my $r (@renderer_modes) {
    foreach my $f (@outfiles) {
        my ($base, $dir, $ext) = fileparse($f, '\..*$');
        my $proj = "../../test/projects/$base-$r.lua";
        $proj = File::Spec->rel2abs($proj);
        print "writing $proj\n";
        open my $fh, ">", $proj or die "cannot open $proj";
        
        print $fh qq{mmCreateView("test", "View3DGL", "::view")\n};
        print $fh qq{mmCreateModule("MMPLDDataSource", "::dat")\n};

        if ($r =~ /^OSPRay/) {

            print $fh qq{mmCreateModule("BoundingBoxRenderer","::bbox")\n};
            print $fh qq{mmCreateModule("OSPRayToGL","::c2gl")\n};
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
            elsif ($r =~ /^Splats/) {
                print $fh qq{mmSetParamValue("::renderer::renderMode", "Splat")\n};
                print $fh qq{mmSetParamValue("::renderer::splat::alphaScaling", "1.000000")\n};
            }
            elsif ($r =~/^Outline/) {
                print $fh qq{mmSetParamValue("::renderer::renderMode", "Outline")\n};
                print $fh qq{mmSetParamValue("::renderer::outline::width", "2.0")\n};
            }
               
            print $fh qq{mmSetParamValue("::distlight::Direction", "-0.500000;0.500000;0.000000")\n};                   
        }
        print $fh qq{mmCreateCall("MultiParticleDataCall", "::renderer::getdata", "::dat::getData")\n};   

        print $fh qq{mmSetParamValue("::view::camstore::settings", [=[{"centre_offset":[0.0,0.0],"convergence_plane":0.0,"eye":0,"far_clipping_plane":12.97671890258789,"film_gate":[0.0,0.0],"gate_scaling":0,"half_aperture_angle":0.2617993950843811,"half_disparity":0.02500000037252903,"image_tile":[0,720,1280,0],"near_clipping_plane":0.012976719066500664,"orientation":[0.17020022869110107,-0.24738462269306183,-0.0711577907204628,-0.9511932134628296],"position":[4.224766731262207,3.3975491523742676,7.757389545440674],"projection_type":0,"resolution_gate":[1280,720]}]=])\n};
        print $fh qq{mmSetParamValue("::view::camstore::autoLoadSettings", "true")\n};
        print $fh qq{mmSetParamValue("::dat::filename", [[$f]])\n};

        print $fh qq{mmSetGUIVisible(false)\n};
        print $fh qq{mmRenderNextFrame()\n};
        print $fh qq{mmScreenshot([[$proj.png]])\n};
        print $fh qq{mmQuit()\n};

        close $fh;

        print $batch qq{megamol.exe $proj\n};
    }
}

close $batch;
