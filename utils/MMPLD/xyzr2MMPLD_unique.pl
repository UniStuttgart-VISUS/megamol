# convert (partnumber) xyzr \times (partnumber) binaries to MMPLD

use strict;
use warnings qw(FATAL);
use File::stat;

use MMPLD;

$#ARGV == 1 or die "usage: $0 <inputfile> <outputfile>";

my $inname = $ARGV[0];
my $outname = $ARGV[1];

print "in: $inname\tout: $outname\n";

open my $infile, '<:raw', $inname or die "could not open input file $!";
my $stat = stat($infile);

my $bytes_read;
my $bytes;
my $size_t_length = 8;

$bytes_read = read $infile, $bytes, $size_t_length;
$bytes_read == $size_t_length or die "could not read number of particles ($size_t_length bytes) $!";

my $numparticles = unpack 'Q', $bytes;
print "expected number of particles: $numparticles\n";
my $filesize = 4 * 4 * $numparticles + $size_t_length;
$filesize == $stat->size or die "file does not meet size expectations ($filesize)";

my $m = MMPLD->new({filename=>$outname, numframes=>1});
$m->StartFrame({frametime=>0.0, numlists=>1});

my %uniqueparts;

# grab radius of first particle
$bytes_read = read $infile, $bytes, 4 * 4;
my ($x, $y, $z, $r) = unpack 'f f f f', $bytes;
$uniqueparts{$bytes} = 1;

for (my $x = 1; $x < $numparticles; $x++) {
    $bytes_read = read $infile, $bytes, 4 * 4;
    $uniqueparts{$bytes} = 1;
}

my $remainingparts = (scalar keys %uniqueparts);
print($remainingparts . " unique particles remaining\n");
$m->StartList({
            vertextype=>$VERTEX_XYZ_FLOAT, colortype=>$COLOR_NONE, globalradius=>$r,
            globalcolor=>"0 192 255 255",
            minintensity=>0.0, maxintensity=>255, particlecount=>$remainingparts
            });

#for my $key (keys %uniqueparts) {
while (my ($key, $value) = each %uniqueparts) {
    my ($x, $y, $z, $r) = unpack 'f f f f', $key;
    $m->AddParticle({
        x=>$x, y=>$y, z=>$z
    });
}

$m->Close();
