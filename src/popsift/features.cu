/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "features.h"
#include "sift_extremum.h"

#include <math_constants.h>

#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

namespace popsift {

/*************************************************************
 * FeaturesBase
 *************************************************************/

FeaturesBase::FeaturesBase( )
    : _num_ext( 0 )
    , _num_ori( 0 )
{ }

FeaturesBase::~FeaturesBase( ) = default;

/*************************************************************
 * FeaturesHost
 *************************************************************/

FeaturesHost::FeaturesHost( )
    : _ext( nullptr )
    , _ori( nullptr )
{ }

FeaturesHost::FeaturesHost( int num_ext, int num_ori )
    : _ext( nullptr )
    , _ori( nullptr )
{
    reset( num_ext, num_ori );
}

FeaturesHost::~FeaturesHost( )
{
    memalign_free( _ext );
    memalign_free( _ori );
}

void FeaturesHost::reset( int num_ext, int num_ori )
{
    if( _ext != nullptr ) { free( _ext ); _ext = nullptr; }
    if( _ori != nullptr ) { free( _ori ); _ori = nullptr; }
    if( _obj != nullptr ) { free( _ori ); _ori = nullptr; }


    _ext = (Feature*)memalign( getPageSize(), num_ext * sizeof(Feature) );
    if( _ext == nullptr ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ext << " features" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }

    _ori = (Descriptor*)memalign( getPageSize(), num_ori * sizeof(Descriptor) );

    if( _ori == nullptr ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ori << " descriptors" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }

    _obj = (float*)memalign( getPageSize(), num_ori * sizeof(float) );

    if( _obj == nullptr ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << " Scene popints" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }




    _numGoodMatches= (int*)memalign( getPageSize(),  sizeof(int) );


    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );
}

void FeaturesHost::pin( )
{
    cudaError_t err;
    err = cudaHostRegister( _ext, getFeatureCount() * sizeof(Feature), 0 );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
             << "    Failed to register feature memory in CUDA." << endl
             << "    Features count: " << getFeatureCount() << endl
             << "    Memory size requested: " << getFeatureCount() * sizeof(Feature) << endl
             << "    " << cudaGetErrorString(err) << endl;
    }
    err = cudaHostRegister( _ori, getDescriptorCount() * sizeof(Descriptor), 0 );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
             << "    Failed to register descriptor memory in CUDA." << endl
             << "    Descriptors count: " << getDescriptorCount() << endl
             << "    Memory size requested: " << getDescriptorCount() * sizeof(Descriptor) << endl
             << "    " << cudaGetErrorString(err) << endl;
    }

    err = cudaHostRegister( _obj, getDescriptorCount() * sizeof(float), 0 );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
             << "    Failed to register Object keypoints in Cuda." << endl
             << "    Descriptors count: " << getDescriptorCount() << endl
             << "    Memory size requested: " << getDescriptorCount() * sizeof(float) << endl
             << "    " << cudaGetErrorString(err) << endl;
    }


    err = cudaHostRegister( _numGoodMatches,sizeof(int), 0 );

}

void FeaturesHost::unpin( )
{
    cudaHostUnregister( _ext );
    cudaHostUnregister( _ori );
    cudaHostUnregister( _obj );
    cudaHostUnregister( _numGoodMatches );


}

void FeaturesHost::print( std::ostream& ostr, bool write_as_uchar ) const
{
    for( int i=0; i<size(); i++ ) {
        _ext[i].print( ostr, write_as_uchar );
    }
}

std::ostream& operator<<( std::ostream& ostr, const FeaturesHost& feature )
{
    feature.print( ostr, false );
    return ostr;
}

/*************************************************************
 * FeaturesDev
 *************************************************************/

FeaturesDev::FeaturesDev( )
    : _ext( nullptr )
    , _ori( nullptr )
    , _rev( nullptr )
    , _obj(nullptr)
    , _numGoodMatches(nullptr)

{

}

FeaturesDev::FeaturesDev( int num_ext, int num_ori )
    : _ext( nullptr )
    , _ori( nullptr )
    , _rev( nullptr )
    , _obj(nullptr)
    , _numGoodMatches(nullptr)

{
    reset( num_ext, num_ori );
}

FeaturesDev::~FeaturesDev( )
{
    cudaFree( _ext );
    cudaFree( _ori );
    cudaFree( _rev );
    cudaFree( _obj );
    cudaFree( _numGoodMatches );

}

void FeaturesDev::reset( int num_ext, int num_ori )
{
    if( _ext != nullptr ) { cudaFree( _ext ); _ext = nullptr; }
    if( _ori != nullptr ) { cudaFree( _ori ); _ori = nullptr; }
    if( _rev != nullptr ) { cudaFree( _rev ); _rev = nullptr; }
    if( _obj != nullptr ) { cudaFree( _obj ); _obj = nullptr; }
    if( _numGoodMatches != nullptr ) { cudaFree( _numGoodMatches ); _numGoodMatches = nullptr; }


    _ext = popsift::cuda::malloc_devT<Feature>   ( num_ext, __FILE__, __LINE__ );
    _ori = popsift::cuda::malloc_devT<Descriptor>( num_ori, __FILE__, __LINE__ );
    _rev = popsift::cuda::malloc_devT<int>       ( num_ori, __FILE__, __LINE__ );
    _obj = popsift::cuda::malloc_devT<float>( num_ori, __FILE__, __LINE__ );

    _numGoodMatches = popsift::cuda::malloc_devT<int>       ( 1, __FILE__, __LINE__ );

    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );

    // resetGoodMatches();


}

__device__ inline float
l2_in_t0( const float4* lptr, const float4* rptr )
{
    const float4  lval = lptr[threadIdx.x];
    const float4  rval = rptr[threadIdx.x];
    const float4  mval = make_float4( lval.x - rval.x,
			              lval.y - rval.y,
			              lval.z - rval.z,
			              lval.w - rval.w );
    float   res = mval.x * mval.x
	        + mval.y * mval.y
	        + mval.z * mval.z
	        + mval.w * mval.w;
    res += shuffle_down( res, 16 );
    res += shuffle_down( res,  8 );
    res += shuffle_down( res,  4 );
    res += shuffle_down( res,  2 );
    res += shuffle_down( res,  1 );
    return res;
}

__global__ void
compute_distance( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len )
{
    if( blockIdx.x >= l_len ) return;
    const int idx = blockIdx.x;

    float match_1st_val = CUDART_INF_F;
    float match_2nd_val = CUDART_INF_F;
    int   match_1st_idx = 0;
    int   match_2nd_idx = 0;

    const float4* lptr = (const float4*)( &l[idx] );

    for( int i=0; i<r_len; i++ )
    {
        const float4* rptr = (const float4*)( &r[i] );

        const float   res  = l2_in_t0( lptr, rptr );

        if( threadIdx.x == 0 )
        {
            if( res < match_1st_val )
            {
                match_2nd_val = match_1st_val;
                match_2nd_idx = match_1st_idx;
                match_1st_val = res;
                match_1st_idx = i;
            }
            else if( res < match_2nd_val )
            {
                match_2nd_val = res;
                match_2nd_idx = i;
            }
        }
        __syncthreads();
    }

    if( threadIdx.x == 0 )
    {
        bool accept = ( match_1st_val / match_2nd_val < 0.8f );
        match_matrix[blockIdx.x] = make_int3( match_1st_idx, match_2nd_idx, accept );
    }

}

__global__ void
show_distance( int3*       match_matrix,
               Feature*    l_ext,
               Descriptor* l_ori,
               int*        l_fem,
               int         l_len,
               Feature*    r_ext,
               Descriptor* r_ori,
               int*        r_fem,
               int         r_len,
               int*         var,
               float*      obj,
               int*        numGoodMatches
             )
{

  *numGoodMatches=0;

    for( int i=0; i<l_len; i++ )
    {
        const float4* lptr  = (const float4*)( &l_ori[i] );
        const float4* rptr1 = (const float4*)( &r_ori[match_matrix[i].x] );
        const float4* rptr2 = (const float4*)( &r_ori[match_matrix[i].y]);


        const float4* lFeatptr  = (const float4*)( &l_ext[i] );

      	float d1 = l2_in_t0( lptr, rptr1 );
      	float d2 = l2_in_t0( lptr, rptr2 );

        const float ratio_thresh = 0.45f;

      	if( threadIdx.x == 0 )
              {
                  if( match_matrix[i].z &&d1<ratio_thresh*d2 )
                      {


                         obj[*numGoodMatches]=l_ext[l_fem[i]].xpos; obj[*numGoodMatches+1]=l_ext[l_fem[i]].ypos;
                         obj[*numGoodMatches+2]=r_ext[r_fem[match_matrix[i].x]].xpos; obj[*numGoodMatches+3]=r_ext[r_fem[match_matrix[i].x]].ypos;



                        //  printf( "\naccept feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                        //        l_fem[i], i,
                        //        r_fem[match_matrix[i].x], match_matrix[i].x,
                        //        r_fem[match_matrix[i].y], match_matrix[i].y,
                        //        d1, d2 );
                        //
                        //
                        // printf("\nKeypoint feat %f %f %f %f ",obj[*numGoodMatches],obj[*numGoodMatches+1],obj[*numGoodMatches+2],obj[*numGoodMatches+3]);

                        *(numGoodMatches)=*(numGoodMatches)+4;
                        // numGoodMatches[0]=12;
                        //
                        // printf("\n Number of matches %d",numGoodMatches[0]);



                        // _obj.push_back(cv::Point2f(l_ext[l_fem[i]].xpos,l_ext[l_fem[i]].ypos));
                        }
      	    else
                      ;// printf( "reject feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                      //         l_fem[i], i,
                      //         r_fem[match_matrix[i].x], match_matrix[i].x,
                      //         r_fem[match_matrix[i].y], match_matrix[i].y,
                      //         d1, d2 );
              }
              __syncthreads();
          }




}

void FeaturesDev::match( FeaturesDev* other)
{
    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );
    // resetGoodMatches();

      int3* match_matrix = popsift::cuda::malloc_devT<int3>( l_len, __FILE__, __LINE__ );

    dim3 grid;
    grid.x = l_len;
    grid.y = 1;
    grid.z = 1;
    dim3 block;
    block.x = 32;
    block.y = 1;
    block.z = 1;

    compute_distance
        <<<grid,block>>>
        ( match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );

    POP_SYNC_CHK;

// Original

    // show_distance
    //     <<<1,32>>>
    //     ( match_matrix,
    //       getFeatures(),
    //       getDescriptors(),
    //       getReverseMap(),
    //       l_len,
    //       other->getFeatures(),
    //       other->getDescriptors(),
    //       other->getReverseMap(),
    //       r_len );

          show_distance
              <<<1,32>>>
              ( match_matrix,
                getFeatures(),
                getDescriptors(),
                getReverseMap(),
                l_len,
                other->getFeatures(),
                other->getDescriptors(),
                other->getReverseMap(),
                r_len,
                _var,
                _obj,
                _numGoodMatches);

    POP_SYNC_CHK;


    cudaFree( match_matrix );
}

void FeaturesDev::clearStructs()
{

  if( _ext != nullptr ) { cudaFree( _ext ); _ext = nullptr; }
  if( _ori != nullptr ) { cudaFree( _ori ); _ori = nullptr; }
  if( _rev != nullptr ) { cudaFree( _rev ); _rev = nullptr; }
  if( _obj != nullptr ) { cudaFree( _obj ); _obj = nullptr; }
  if( _numGoodMatches != nullptr ) { cudaFree( _numGoodMatches ); _numGoodMatches = nullptr; }

}



/*************************************************************
 * Feature
 *************************************************************/

void Feature::print( std::ostream& ostr, bool write_as_uchar ) const
{
    float sigval =  1.0f / ( sigma * sigma );

    for( int ori=0; ori<num_ori; ori++ ) {
        ostr << xpos << " " << ypos << " "
             << sigval << " 0 " << sigval << " ";
        if( write_as_uchar ) {
            for( int i=0; i<128; i++ ) {
                ostr << roundf(desc[ori]->features[i]) << " ";
            }
        } else {
            ostr << std::setprecision(3);
            for( int i=0; i<128; i++ ) {
                ostr << desc[ori]->features[i] << " ";
            }
            ostr << std::setprecision(6);
        }
        ostr << std::endl;
    }
}

std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
{
    feature.print( ostr, false );
    return ostr;
}

} // namespace popsift
