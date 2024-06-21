#pragma once
#include "CUDA_types.cuh" 

__global__ void cu_rotCVVoxel_Streamed_16x16(
	int nVoxelX, int nCVVoxel, int nYPR,
	int yprID, 
	float* dev_m4m,  float* dev_CVVoxels,  
		CU_SLOT_BUFFER_TYPE* dev_nPixel, CU_SLOT_BUFFER_TYPE* dev_pType, CU_SLOT_BUFFER_TYPE* dev_pZcrd, CU_SLOT_BUFFER_TYPE* dev_pZnrm);

