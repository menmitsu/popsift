/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "sift_pyramid.h"
#include "sift_octave.h"
#include "sift_extremum.h"
#include "common/plane_2d.h"

__global__
void ext_desc_iloop( const int           octave,
                     cudaTextureObject_t layer_tex,
                     const int           width,
                     const int           height );

namespace popsift
{

inline static bool start_ext_desc_iloop( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;
    grid.x = hct.ori_ct[octave];
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = 1;
    block.z = 16;

    ext_desc_iloop
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexLinear( ),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    return true;
}

}; // namespace popsift

