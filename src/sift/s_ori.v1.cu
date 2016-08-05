#include "sift_pyramid.h"
#include "sift_constants.h"
#include "sift_extrema_mgmt.h"
#include "s_gradiant.h"
#include "excl_blk_prefix_sum.h"
#include "debug_macros.h"

#include <math.h>
#include <stdio.h>
#include <inttypes.h>

using namespace popart;
using namespace std;

#undef V2_WITH_VLFEAT_SMOOTHING
#define V2_WITH_OPENCV_SMOOTHING

/*************************************************************
 * V1: device side
 *************************************************************/

__device__
inline float compute_angle( int bin, float hc, float hn, float hp )
{
    /* interpolate */
    float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);

    /* clamp */
    di = (di < 0) ? 
            (di + ORI_NBINS) : 
            ((di >= ORI_NBINS) ? (di - ORI_NBINS) : (di));

    float th = __fdividef( M_PI2 * di, ORI_NBINS ) - M_PI;
    // float th = ((M_PI2 * di) / ORI_NBINS);
    return th;
}

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 * direct curve fitting approach
 */
__global__
void compute_keypoint_orientations( Extremum*     extremum,
                                       int*          extrema_counter,
                                       Plane2D_float layer )
{
    uint32_t w   = layer.getWidth();
    uint32_t h   = layer.getHeight();

    // if( threadIdx.y >= mgmt->getCounter() ) return;

    Extremum* ext = &extremum[blockIdx.x];

    float hist[ORI_NBINS];
    for (int i = 0; i < ORI_NBINS; i++) hist[i] = 0.0f;

    /* keypoint fractional geometry */
    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;

    /* orientation histogram radius */
    float  sigw = ORI_WINFACTOR * sig;
    int32_t rad  = (int)rintf((3.0f * sigw));

    float factor = __fdividef( -0.5f, (sigw * sigw) );
    int sq_thres  = rad * rad;

    int32_t xmin = max(1,     (int32_t)floor(x - rad));
    int32_t xmax = min(w - 2, (int32_t)floor(x + rad));
    int32_t ymin = max(1,     (int32_t)floor(y - rad));
    int32_t ymax = min(h - 2, (int32_t)floor(y + rad));

    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy;

    for(int i = threadIdx.x; i < loops; i+=ORI_V1_NUM_THREADS)
    {
        int yy = i / wx + ymin;
        int xx = i % wx + xmin;

        float grad;
        float theta;
        get_gradiant( grad,
                      theta,
                      xx,
                      yy,
                      layer );

        float dx = xx - x;
        float dy = yy - y;

        int sq_dist  = dx * dx + dy * dy;
        if (sq_dist <= sq_thres) {
            float weight = grad * expf(sq_dist * factor);

            int bidx = (int)rintf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );
            // int bidx = (int)roundf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );

            if( bidx > ORI_NBINS ) {
                printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
            }

            bidx = (bidx == ORI_NBINS) ? 0 : bidx;

            hist[bidx] += weight;
        }
    }

    /* reduction here */
    for (int i = 0; i < ORI_NBINS; i++) {
        hist[i] += __shfl_down( hist[i], 8 );
        hist[i] += __shfl_down( hist[i], 4 );
        hist[i] += __shfl_down( hist[i], 2 );
        hist[i] += __shfl_down( hist[i], 1 );
        hist[i]  = __shfl( hist[i], 0 );
    }


    if(threadIdx.x != 0) return;

    float xcoord[ORI_NBINS];
    float yval[ORI_NBINS];

    int   maxbin[ORIENTATION_MAX_COUNT];
    float y_max[ORIENTATION_MAX_COUNT];

    #pragma unroll
    for( int i=0; i<ORIENTATION_MAX_COUNT; i++ ) {
        maxbin[i] = -1;
        y_max[i] = -INFINITY;
    }

#ifdef V2_WITH_VLFEAT_SMOOTHING
    for( int i=0; i<3; i++ ) {
        for(int bin = 0; bin < ORI_NBINS; bin++) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            xcoord[bin] = ( hist[prev] + hist[bin] + hist[next] ) / 3.0f;
        }
        for(int bin = 0; bin < ORI_NBINS; bin++) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            hist[bin] = ( xcoord[prev] + xcoord[bin] + xcoord[next] ) / 3.0f;
        }
    }
#endif // V2_WITH_VLFEAT_SMOOTHING

#ifdef V2_WITH_OPENCV_SMOOTHING
    for(int bin = 0; bin < ORI_NBINS; bin++) {
        int prev2 = bin - 2;
        int prev1 = bin - 1;
        int next1 = bin + 1;
        int next2 = bin + 2;
        if( prev2 < 0 )          prev2 += ORI_NBINS;
        if( prev1 < 0 )          prev1 += ORI_NBINS;
        if( next1 >= ORI_NBINS ) next1 -= ORI_NBINS;
        if( next2 >= ORI_NBINS ) next2 -= ORI_NBINS;
        xcoord[bin] = (   hist[prev2] + hist[next2]
                      + ( hist[prev1] + hist[next1] ) * 4.0f
                      +   hist[bin] * 6.0f ) / 16.0f;
    }
    for(int bin = 0; bin < ORI_NBINS; bin++) {
        hist[bin] = xcoord[bin];
    }
#endif // V2_WITH_OPENCV_SMOOTHING

    for(int bin = 0; bin < ORI_NBINS; bin++) {
        // int prev = bin - 1;
        // if( prev < 0 ) prev = ORI_NBINS - 1;
        // int next = bin + 1;
        // if( next == ORI_NBINS ) next = 0;
        int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
        int next = bin == ORI_NBINS-1 ? 0 : bin+1;

        if( hist[bin] > max( hist[prev], hist[next] ) ) {
            const float num = 3.0f * hist[prev] - 4.0f * hist[bin] + hist[next];
            const float denB = 2.0f * ( hist[prev] - 2.0f * hist[bin] + hist[next] );

            float newbin = __fdividef( num, denB ); // * M_PI/18.0f; // * 10.0f;
            if( newbin >= 0 && newbin <= 2 ) {
                xcoord[bin] = prev + newbin;
                yval[bin]   = -(num*num) / (4.0f * denB) + hist[prev];

#ifdef LOWE_ORIENTATION_MAX
                if( yval[bin] > y_max[0] ) {
                    y_max[2]  = y_max[1];
                    y_max[1]  = y_max[0];
                    y_max[0]  = yval[bin];
                    maxbin[2] = maxbin[1];
                    maxbin[1] = maxbin[0];
                    maxbin[0] = bin;
                } else if( yval[bin] > y_max[1] ) {
                    y_max[2]  = y_max[1];
                    y_max[1]  = yval[bin];
                    maxbin[2] = maxbin[1];
                    maxbin[1] = bin;
                } else if( yval[bin] > y_max[2] ) {
                    y_max[2]  = yval[bin];
                    maxbin[2] = bin;
                }
#else // LOWE_ORIENTATION_MAX
                if( yval[bin] > y_max[0] ) {
                    y_max[3]  = y_max[2];
                    y_max[2]  = y_max[1];
                    y_max[1]  = y_max[0];
                    y_max[0]  = yval[bin];
                    maxbin[3] = maxbin[2];
                    maxbin[2] = maxbin[1];
                    maxbin[1] = maxbin[0];
                    maxbin[0] = bin;
                } else if( yval[bin] > y_max[1] ) {
                    y_max[3]  = y_max[2];
                    y_max[2]  = y_max[1];
                    y_max[1]  = yval[bin];
                    maxbin[3] = maxbin[2];
                    maxbin[2] = maxbin[1];
                    maxbin[1] = bin;
                } else if( yval[bin] > y_max[2] ) {
                    y_max[3]  = y_max[2];
                    y_max[2]  = yval[bin];
                    maxbin[3] = maxbin[2];
                    maxbin[2] = bin;
                } else if( yval[bin] > y_max[3] ) {
                    y_max[3]  = yval[bin];
                    maxbin[3] = bin;
                }
#endif // LOWE_ORIENTATION_MAX
            }
        }
    }

    float chosen_bin = xcoord[maxbin[0]];
    if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;

    float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS) - M_PI;

    ext->orientation[0] = th;
    int angles = 1;

    for( int i=1; i<ORIENTATION_MAX_COUNT; i++ ) {
        if( y_max[i] < -1000.0f ) break; // this is a random number: no orientation can be this small

        if( y_max[i] < 0.8f * y_max[0] ) break;

        float chosen_bin = xcoord[maxbin[i]];
        if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;
        float th = __fdividef(M_PI2 * chosen_bin, ORI_NBINS) - M_PI;

        ext->orientation[i] = th;
        angles++;
    }

    ext->num_ori = angles;
}

class ExtremaRead
{
    const Extremum* const _oris;
public:
    inline __device__
    ExtremaRead( const Extremum* const d_oris ) : _oris( d_oris ) { }

    inline __device__
    int get( int n ) const { return _oris[n].num_ori; }
};

class ExtremaWrt
{
    Extremum* _oris;
public:
    inline __device__
    ExtremaWrt( Extremum* d_oris ) : _oris( d_oris ) { }

    inline __device__
    void set( int n, int value ) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int* _extrema_counter;
public:
    inline __device__
    ExtremaTot( int* extrema_counter ) : _extrema_counter( extrema_counter ) { }

    inline __device__
    void set( int value ) { *_extrema_counter = value; }
};

class ExtremaWrtMap
{
    int* _featvec_to_extrema_mapper;
    int  _max_feat;
public:
    inline __device__
    ExtremaWrtMap( int* featvec_to_extrema_mapper, int max_feat )
        : _featvec_to_extrema_mapper( featvec_to_extrema_mapper )
        , _max_feat( max_feat )
    { }

    inline __device__
    void set( int base, int num, int value )
    {
        int* baseptr = &_featvec_to_extrema_mapper[base];
        do {
            num--;
            if( base + num < _max_feat ) {
                baseptr[num] = value;
            }
        } while( num > 0 );
    }
};

__global__
void ori_prefix_sum( int*      extrema_counter,
                     int*      featvec_counter,
                     Extremum* extremum,
                     int*      d_feat_to_ext_map )
{
    ExtremaRead r( extremum );
    ExtremaWrt  w( extremum );
    ExtremaTot  t( featvec_counter );
    ExtremaWrtMap wrtm( d_feat_to_ext_map, d_max.orientations );
    ExclusivePrefixSum::Block<ExtremaRead,ExtremaWrt,ExtremaTot,ExtremaWrtMap>( *extrema_counter, r, w, t, wrtm );

    __syncthreads();

    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        *featvec_counter = min( *featvec_counter, d_max.orientations );

        // printf("Leave %s, %d extrema -> %d oris\n", __func__, *extrema_counter, *featvec_counter );
    }
}

#if 0
// verify that mapping from feature index to extremum index works

__global__
void print_fmap( int       octave,
                 int       level,
                 int*      extrema_counter,
                 int*      featvec_counter,
                 Extremum* extremum,
                 int*      d_feat_to_ext_map )
{
    if( *extrema_counter == 0 && *featvec_counter == 0 ) return;

    printf( "o/l %d/%d #e:%d #o:%d ", octave, level, *extrema_counter, *featvec_counter );
    for( int i=0; i<*featvec_counter; i++ ) {
        if( i != 0 ) printf(",  " );
        if( i>0 && i % 8 == 0 )
            printf("\n       ");
        int e = d_feat_to_ext_map[i];
        int n = extremum[e].num_ori;
        if( n != 1 )
            printf( "%2d->%2d (%d)", i, e, extremum[e].num_ori );
        else
            printf( "%2d->%2d", i, e );
    }
    printf("\n");
}
#endif


/*************************************************************
 * V4: host side
 *************************************************************/

__global__
void orientation_starter( Extremum*     extremum,
                          int*          extrema_counter,
                          int*          featvec_counter,
                          int*          d_feat_to_ext_map,
                          Plane2D_float layer )
{
#ifdef USE_DYNAMIC_PARALLELISM // defined in_s_pyramid.h
    dim3 block;
    dim3 grid;
    grid.x  = *extrema_counter;
    block.x = ORI_V1_NUM_THREADS;

    if( grid.x != 0 ) {
        compute_keypoint_orientations
            <<<grid,block>>>
            ( extremum,
              extrema_counter,
              layer );

        block.x = 32;
        block.y = 32;
        grid.x  = 1;
        ori_prefix_sum
            <<<grid,block>>>
            ( extrema_counter,
              featvec_counter,
              extremum,
              d_feat_to_ext_map );
    }
#endif // USE_DYNAMIC_PARALLELISM
}

#ifdef USE_DYNAMIC_PARALLELISM // defined in_s_pyramid.h
__host__
void Pyramid::orientation_v1( )
{
    cerr << "Calling ori with dynamic parallelism" << endl;

    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        // int*  orientation_num_blocks = oct_obj.getNumberOfOriBlocks( );

        for( int level=1; level<_levels-2; level++ ) {
            cudaStream_t oct_str = oct_obj.getStream(level+2);

            orientation_starter
                <<<1,1,0,oct_str>>>
                ( oct_obj.getExtrema( level ),
                  oct_obj.getExtremaCtPtrD( level ),
                  oct_obj.getFeatVecCtPtrD( level ),
                  oct_obj.getFeatToExtMapD( level ),
                  oct_obj.getData( level ) );
        }
    }

#if 0
    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-2; level++ ) {
            cudaDeviceSynchronize();
            Octave& oct_obj = _octaves[octave];
            int* extrema_counters  = oct_obj.getExtremaCounterD( );
            int* extrema_counter   = &extrema_counters[level];
            int* featvec_counters  = oct_obj.getFeatVecCounterD( );
            int* featvec_counter   = &featvec_counters[level];
            print_fmap
                <<<1,1>>>
                ( octave,
                  level,
                  extrema_counter,
                  featvec_counter,
                  oct_obj.getExtrema( level ),
                  oct_obj.getFeatToExtMapD( level ) );
        }
    }
    cudaDeviceSynchronize();
#endif
}

#else // not USE_DYNAMIC_PARALLELISM

__host__
void Pyramid::orientation_v1( )
{
    cerr << "Calling ori with -no- dynamic parallelism" << endl;

    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        for( int level=3; level<_levels; level++ ) {
            cudaStreamSynchronize( oct_obj.getStream(level) );
        }

        oct_obj.readExtremaCount( );
        cudaDeviceSynchronize( );

        int* orientation_num_blocks = oct_obj.getNumberOfOriBlocks( );

        for( int level=1; level<_levels-2; level++ ) {
            cudaStream_t oct_str = oct_obj.getStream(level+2);

            dim3 block;
            dim3 grid;
            grid.x  = oct_obj.getExtremaCtPtrH( level );
            block.x = ORI_V1_NUM_THREADS;

            if( grid.x != 0 ) {
                compute_keypoint_orientations
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getExtrema( level ),
                      oct_obj.getExtremaCtPtrD( level ),
                      oct_obj.getData( level ),
                      &orientation_num_blocks[level],
                      grid.x * grid.y );

                block.x = 32;
                block.y = 32;
                grid.x  = 1;
                ori_prefix_sum
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getExtremaCtPtrD( level ),
                      oct_obj.getFeatVecMCtPtrD( level ),
                      oct_obj.getExtrema( level ),
                      oct_obj.getFeatToExtMapD(level) );
            }
        }
    }

#if 0
    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-2; level++ ) {
            cudaDeviceSynchronize();
            Octave& oct_obj = _octaves[octave];
            int* extrema_counters  = oct_obj.getExtremaCounterD( );
            int* extrema_counter   = &extrema_counters[level];
            int* featvec_counters  = oct_obj.getFeatVecCounterD( );
            int* featvec_counter   = &featvec_counters[level];
            print_fmap
                <<<1,1>>>
                ( octave,
                  level,
                  extrema_counter,
                  featvec_counter,
                  oct_obj.getExtrema( level ),
                  oct_obj.getFeatToExtMapD( level ) );
        }
    }
    cudaDeviceSynchronize();
#endif
}
#endif // not USE_DYNAMIC_PARALLELISM

