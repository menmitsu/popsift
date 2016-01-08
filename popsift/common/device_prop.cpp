#include <iostream>

#include "device_prop.h"
#include "debug_macros.h"

using namespace std;

device_prop_t::device_prop_t( )
{
    cudaError_t err;

    err = cudaGetDeviceCount( &_num_devices );
    POP_CUDA_FATAL_TEST( err, "Cannot count devices" );

    for( int n=0; n<_num_devices; n++ ) {
        cudaDeviceProp* p;
        _properties.push_back( p = new cudaDeviceProp );
        err = cudaGetDeviceProperties( p, n );
        POP_CUDA_FATAL_TEST( err, "Cannot get properties for a device" );
    }
    err = cudaSetDevice( 0 );
    POP_CUDA_FATAL_TEST( err, "Cannot set device 0" );
}

void device_prop_t::print( )
{
    // for( auto ptr : _properties ) {
    std::vector<cudaDeviceProp*>::const_iterator p;
    for( p = _properties.begin(); p!=_properties.end(); p++ ) {
        cudaDeviceProp* ptr = *p;
        std::cout << "Device information:" << endl
                  << "    Name: " << ptr->name << endl
                  << "    Compute Capability:    " << ptr->major << "." << ptr->minor << endl
                  << "    Total device mem:      " << ptr->totalGlobalMem << " B "
                  << ptr->totalGlobalMem/1024 << " kB "
                  << ptr->totalGlobalMem/(1024*1024) << " MB " << endl
                  << "    Per-block shared mem:  " << ptr->sharedMemPerBlock << endl
                  << "    Warp size:             " << ptr->warpSize << endl
                  << "    Max threads per block: " << ptr->maxThreadsPerBlock << endl
                  << "    Max threads per SM(X): " << ptr->maxThreadsPerMultiProcessor << endl
                  << "    Max block sizes:       "
                  << "{" << ptr->maxThreadsDim[0]
                  << "," << ptr->maxThreadsDim[1]
                  << "," << ptr->maxThreadsDim[2] << "}" << endl
                  << "    Max grid sizes:        "
                  << "{" << ptr->maxGridSize[0]
                  << "," << ptr->maxGridSize[1]
                  << "," << ptr->maxGridSize[2] << "}" << endl
                  << "    Number of SM(x)s:      " << ptr->multiProcessorCount << endl
                  << "    Concurrent kernels:    " << (ptr->concurrentKernels?"yes":"no") << endl
                  << "    Mapping host memory:   " << (ptr->canMapHostMemory?"yes":"no") << endl
                  << "    Unified addressing:    " << (ptr->unifiedAddressing?"yes":"no") << endl
                  << endl;
    }
}

void device_prop_t::set( int n )
{
    cudaError_t err;
    err = cudaSetDevice( n );
    POP_CUDA_FATAL_TEST( err, "Cannot set device 0" );
}

device_prop_t::~device_prop_t( )
{
    // for( auto ptr : _properties ) {
    std::vector<cudaDeviceProp*>::const_iterator p;
    for( p = _properties.begin(); p!=_properties.end(); p++ ) {
        cudaDeviceProp* ptr = *p;
        delete ptr;
    }
}

