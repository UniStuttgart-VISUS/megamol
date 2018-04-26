use strict;
use warnings qw(FATAL);
use Cwd;
use MMPLD;

my $outfile;
my $m;
my $numPoints = 4;

my @outfiles = ();

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

$outfile = "test_xyz_float_rgb_float.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_RGB_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_rgb_float.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_RGB_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_rgba_float.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_RGBA_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_rgba_float.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_RGBA_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_rgba_byte.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_RGBA_BYTE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(255.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_rgba_byte.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_RGBA_BYTE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(255.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_int_float.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_int_float.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyz_float_none.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints, globalcolor=>"255 255 0 255"
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

$outfile = "test_xyzr_float_none.mmpld";
push @outfiles, $outfile;
$m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

$m->StartList({
            vertextype=>$VERTEX_XYZR_FLOAT, colortype=>$COLOR_NONE, globalradius=>0.1,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints, globalcolor=>"255 255 0 255"
            });
AddParticles(1.0);
$m->OverrideBBox(-2,-2,-2,2,2,2);
$m->Close();

open my $batch, ">", "SphereTest.bat" or die "cannot open batch file";

my @renderers = ("SimpleSphereRenderer", "SimpleGeoSphereRenderer", "NGSphereRenderer");
foreach my $r (@renderers) {
    foreach my $f (@outfiles) {
        my $proj = "$f-$r.lua";
        open my $fh, ">", $proj or die "cannot open $proj";
        print $fh qq{mmCreateView("test", "View3D", "::v")\n};
        print $fh qq{mmCreateJob("imagemaker", "ScreenShooter", "::imgmaker")\n};
        print $fh qq{mmCreateModule("$r", "::rnd")\n};
        print $fh qq{mmCreateModule("MMPLDDataSource", "::dat")\n};
        print $fh qq{mmCreateCall("CallRender3D", "::v::rendering", "::rnd::rendering")\n};
        print $fh qq{mmCreateCall("MultiParticleDataCall", "::rnd::getData", "::dat::getData")\n};
        print $fh qq{mmSetParamValue("::dat::filename", "}.getcwd().qq{/$f")\n};
        print $fh qq{mmSetParamValue("::imgmaker::filename", "}.getcwd().qq{/$f-$r.png")\n};
        print $fh qq{mmSetParamValue("::imgmaker::view", "test")\n};
        print $fh qq|mmSetParamValue("::v::camsettings", "{ApertureAngle=30\\nCoordSystemType=2\\nNearClip=14.746623992919921875\\nFarClip=32.5738677978515625\\nProjection=0\\nStereoDisparity=0.300000011920928955078125\\nStereoEye=0\\nFocalDistance=23.6602458953857421875\\nPositionX=14.33369541168212890625\\nPositionY=8.6866302490234375\\nPositionZ=16.7001476287841796875\\nLookAtX=0\\nLookAtY=0\\nLookAtZ=0\\nUpX=-0.1268725097179412841796875\\nUpY=0.920388698577880859375\\nUpZ=-0.3698485195636749267578125\\nTileLeft=0\\nTileBottom=0\\nTileRight=1280\\nTileTop=720\\nVirtualViewWidth=1280\\nVirtualViewHeight=720\\nAutoFocusOffset=0\\n}")\n|;
        print $fh qq{mmSetParamValue("::imgmaker::closeAfter", "true")\n};
        print $fh qq{mmSetParamValue("::imgmaker::trigger", " ")\n};
        close $fh;

        print $batch qq{mmconsole.exe -p ..\\utils\\MMPLD\\$proj\n};
    }
}