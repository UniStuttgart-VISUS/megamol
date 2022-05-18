use strict;
use warnings qw(FATAL);

use MMPLD;

$#ARGV == 0 or die "usage: $0 <outputfile>";

my $outfile = $ARGV[0];

my $m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

my $numPoints = 100000000;

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>1.0,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints + 8
            });

my @xs;
my @ys;
my @zs;
my @is;

my $currPoint = 0;
my $oldperc = 0;

for (my $p = 0; $p < $numPoints; $p++) {
    $m->AddParticle({x=>0, y=>0, z=>$p, i=>1});
}

$m->AddParticle({x=>-100, y=>-100, z=>0, i=>1});
$m->AddParticle({x=> 100, y=>-100, z=>0, i=>1});
$m->AddParticle({x=>-100, y=> 100, z=>0, i=>1});
$m->AddParticle({x=> 100, y=> 100, z=>0, i=>1});
$m->AddParticle({x=>-100, y=>-100, z=>$numPoints - 1, i=>1});
$m->AddParticle({x=> 100, y=>-100, z=>$numPoints - 1, i=>1});
$m->AddParticle({x=>-100, y=> 100, z=>$numPoints - 1, i=>1});
$m->AddParticle({x=> 100, y=> 100, z=>$numPoints - 1, i=>1});
