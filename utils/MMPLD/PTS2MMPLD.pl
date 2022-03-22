use strict;
use warnings qw(FATAL);

use MMPLD;

$#ARGV == 1 or die "usage: $0 <inputfile> <outputfile>";

my $outfile = $ARGV[1];

my $m = MMPLD->new({filename=>$outfile, numframes=>1});
$m->StartFrame({frametime=>1.23, numlists=>1});

my $temp = <>;
$temp =~ /(\d+)/;
my $numPoints = $1;

$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_INTENSITY_FLOAT, globalradius=>1.0,
            minintensity=>0.0, maxintensity=>255, particlecount=>$numPoints
            });

my @xs;
my @ys;
my @zs;
my @is;

my $currPoint = 0;
my $oldperc = 0;

while (<>) {
    my @vals = split /\s/;
    if ($#vals != 6) {
        next;
    }
    #push @xs, $vals[0];
    #push @ys, $vals[1];
    #push @zs, $vals[2];
    #push @is, $vals[3];
    
    $m->AddParticle({
        x=>$vals[0], y=>$vals[1], z=>$vals[2], i=>$vals[3]
    });
    
    #if ($#xs == 1) {
    #    last;
    #}
    $currPoint++;
    my $newperc = int(($currPoint * 100) / $numPoints);
    if ($newperc != $oldperc) {
        print "$newperc%\n";
        $oldperc = $newperc;
    }
    
}

print "got $currPoint points, should be $numPoints: " . (($currPoint==$numPoints)?'Ok':'Problem') . ".\n";
