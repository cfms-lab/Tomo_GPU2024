#pragma once
#include "CUDA_types.cuh" 


//ToDo1: SLOT_CAPACITY->16
//ToDo2: int32 -> uchar8   https://stackoverflow.com/questions/5447570/cuda-atomic-operations-on-unsigned-chars


__global__ void cu_rotVoxel_16x16(
	int nTri, int nYPR, int nVoxelX, int yprID_to_start,
	float* dev_m4m,  float* dev_tri0,  
	CU_SLOT_BUFFER_TYPE* dev_nPixel, CU_SLOT_BUFFER_TYPE* dev_pType, CU_SLOT_BUFFER_TYPE* dev_pZcrd, CU_SLOT_BUFFER_TYPE* dev_pZnrm);


__global__ void cu_rotVoxel_Streamed_16x16(
	int nVoxelX, int nYPR, int nFlatTri, 
	int yprID, int triID_to_start,
	float* dev_m4m,  float* dev_tri0,  
		CU_SLOT_BUFFER_TYPE* dev_nPixel, CU_SLOT_BUFFER_TYPE* dev_pType, CU_SLOT_BUFFER_TYPE* dev_pZcrd, CU_SLOT_BUFFER_TYPE* dev_pZnrm);

