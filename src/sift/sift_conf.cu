#include "sift_conf.h"

namespace popart
{

Config::Config( )
    : start_sampling( -1 )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    , _sift_mode( Config::PopSift )
    , log_mode( Config::None )
    , scaling_mode( Config::IndirectUnfilteredDownscaling )
    , verbose( false )
    , gauss_group_size( 1 )
    , _assume_initial_blur( false )
    , _initial_blur( 0.0f )
    , _bemap_orientation( false )
    , _print_gauss_tables( false )
{
}

void Config::setMode( Config::SiftMode m )
{
    _sift_mode = m;
}

void Config::setVerbose( bool on )
{
    verbose = on;
}

void Config::setLogMode( LogMode mode )
{
    log_mode = mode;
}

void Config::setScalingMode( ScalingMode mode )
{
    scaling_mode = mode;
}

void Config::setDownsampling( float v ) { start_sampling = v; }
void Config::setOctaves( int v ) { octaves = v; }
void Config::setLevels( int v ) { levels = v; }
void Config::setSigma( float v ) { sigma = v; }
void Config::setEdgeLimit( float v ) { _edge_limit = v; }
void Config::setThreshold( float v ) { _threshold = v; }
void Config::setPrintGaussTables() { _print_gauss_tables = true; }

void Config::setInitialBlur( float blur )
{
    _assume_initial_blur = true;
    _initial_blur        = blur;
}

void Config::setGaussGroup( int groupsize )
{
    gauss_group_size = groupsize;
}

int  Config::getGaussGroup( ) const
{
    return gauss_group_size;
}

bool Config::hasInitialBlur( ) const
{
    return _assume_initial_blur;
}

float Config::getInitialBlur( ) const
{
    return _initial_blur;
}

void Config::setBemapOrientation( )
{
    _bemap_orientation = true;
}

bool Config::getBemapOrientation( ) const
{
    return _bemap_orientation;
}

float Config::getPeakThreshold() const
{
    return ( _threshold * 0.5f * 255.0f / levels );
}

bool Config::ifPrintGaussTables() const
{
    return _print_gauss_tables;
}

Config::SiftMode Config::getSiftMode() const
{
    return _sift_mode;
}


}; // namespace popart

