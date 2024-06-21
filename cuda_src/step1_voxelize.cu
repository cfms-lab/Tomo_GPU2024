#include "step1_voxelize.cuh"
#include <iostream>
#include <stdio.h>
#include "atomicWrite.cuh"

using namespace Tomo;

__device__  __inline__ bool  cu_getBaryCoord2D(
	/*inputs*/ CU_FLOAT32* p, CU_FLOAT32* triA, CU_FLOAT32* triB, CU_FLOAT32* triC,
	/*output*/ CU_FLOAT32& u, CU_FLOAT32& v, CU_FLOAT32& w)
{
	CU_FLOAT32 v0[2] = { triB[0] - triA[0], triB[1] - triA[1] };
	CU_FLOAT32 v1[2] = { triC[0] - triA[0], triC[1] - triA[1] };
	CU_FLOAT32 v2[2] = { p[0] - triA[0], p[1] - triA[1] };

	CU_FLOAT32 d00 = cu_dot2D(v0, v0);
	CU_FLOAT32 d01 = cu_dot2D(v0, v1);
	CU_FLOAT32 d11 = cu_dot2D(v1, v1);
	CU_FLOAT32 d20 = cu_dot2D(v2, v0);
	CU_FLOAT32 d21 = cu_dot2D(v2, v1);

	CU_FLOAT32 denom = d00 * d11 - d01 * d01;

	if ( abs(denom) > cu_fMARGIN)
	{
		v = (d11 * d20 - d01 * d21) / denom;
		w = (d00 * d21 - d01 * d20) / denom;
		u = 1.0 - v - w;
		return  (u >= -cu_fMARGIN && v >= -cu_fMARGIN && v <= 1. + cu_fMARGIN && u + v <= 1. + cu_fMARGIN);
	}
	return false;
}

__device__  __inline__ CU_SLOT_BUFFER_TYPE  cu_getBaryCoordZ(
	/*inputs*/ CU_FLOAT32 p_x,CU_FLOAT32 p_y, 
	CU_FLOAT32* triA, CU_FLOAT32* triB, CU_FLOAT32* triC)
{
	CU_FLOAT32 v0[2] = { triB[0] - triA[0], triB[1] - triA[1] };
	CU_FLOAT32 v1[2] = { triC[0] - triA[0], triC[1] - triA[1] };
	CU_FLOAT32 v2[2] = { p_x - triA[0], p_y - triA[1] };

	CU_FLOAT32 d00 = cu_dot2D(v0, v0);
	CU_FLOAT32 d01 = cu_dot2D(v0, v1);
	CU_FLOAT32 d11 = cu_dot2D(v1, v1);
	CU_FLOAT32 d20 = cu_dot2D(v2, v0);
	CU_FLOAT32 d21 = cu_dot2D(v2, v1);

	CU_FLOAT32 denom = d00 * d11 - d01 * d01;

	if ( abs(denom) > cu_fMARGIN)
	{
		CU_FLOAT32 v = (d11 * d20 - d01 * d21) / denom;
		CU_FLOAT32 w = (d00 * d21 - d01 * d20) / denom;
		CU_FLOAT32 u = 1.0 - v - w;
		if(u >= -cu_fMARGIN && v >= -cu_fMARGIN && v <= 1. + cu_fMARGIN && u + v <= 1. + cu_fMARGIN)
		{
			return int(u * triA[2] + v * triB[2] + w * triC[2]);
		}
	}
	return -1;
}




#ifdef _CUDA_USE_ROTATE_AND_PIXELIZE_IN_ONE_STEP

	__global__ void cu_rotVoxel_16x16(
		int nFlatTri, int nYPR, int nVoxelX, int yprID_to_start,
		float* cu_m4x3,	float* cu_flattri0,	
		CU_SLOT_BUFFER_TYPE* cu_nPixel, CU_SLOT_BUFFER_TYPE* cu_pType, CU_SLOT_BUFFER_TYPE* cu_pZcrd, CU_SLOT_BUFFER_TYPE* cu_pZnrm)
	{

		//Note: blockIdx = ( nFlatTri * _nYPRInBatch, 1)
		int triID	= blockIdx.x;//triangle ID within a same block(orientation).
		int wrkID	= blockIdx.y;//ypr data ID within a batch.

		const int thID	= threadIdx.y * blockDim.x + threadIdx.x;//this block is (16,16,1) size.
		int yprID = yprID_to_start  + wrkID;//Caution: m4x3_ID == 0 ~ nYPR-1.
		if (triID >= nFlatTri ||yprID >= nYPR) return;
	
		__shared__ float ftri[CU_FLATTRI_SIZE_16], m4x3[CU_MATRIX_SIZE_12];

		//per-BLOCK operation +++++++++++++++
		if (thID < CU_FLATTRI_SIZE_16)	{//copy tri coord data to shared memory.
			ftri[thID]	= cu_flattri0[triID * CU_FLATTRI_SIZE_16 + thID];
		}
		else if ((thID-CU_FLATTRI_SIZE_16) >=0 && (thID-CU_FLATTRI_SIZE_16) <  + CU_MATRIX_SIZE_12)	{//copy matrix data to shared memory
			m4x3[(thID - CU_FLATTRI_SIZE_16)]	= cu_m4x3[ yprID * CU_MATRIX_SIZE_12 + (thID - CU_FLATTRI_SIZE_16)];
		}
		__syncthreads();

		//matrix operation(YPR rotation + translation of AABB corner to origin)
		if(thID<=3) {
			cu_matrixOp(m4x3, ftri + thID * 3);
		}
		__syncthreads();

		if(thID<=1) {
			ftri[12 + thID] = min(min(ftri[0 + thID], ftri[3+ thID]), ftri[6 + thID]);//global_AABB_x0, _y0
		}
		__syncthreads();

		if(thID<=2) {
			ftri[9+thID] -= m4x3[4*thID+3];//eliminate translation from normal vector
		}
		__syncthreads();
		//+++++++++++++++++++++++++++++++++++


		//per-THREAD operation ------------------------------------
		//  (16,16)내에서의 로컬 픽셀좌표는 (threadIdx.x, threadIdx.y)
		//  global 좌표계는 (AABB_x0 + threadIdx.x, AABB_y0 + threadIdx.y)임.
		//AABB of current triangle
		const register int& global_AABB_x0 = ftri[12];
		const register int& global_AABB_y0 = ftri[13];
		register float vxl_global_cnt[2] = { //global 좌표계에서의 현재 복셀의 위치.
						(global_AABB_x0 + threadIdx.x) + cu_HALF_VOXEL_SIZE,
						(global_AABB_y0 + threadIdx.y) + cu_HALF_VOXEL_SIZE };
		if (vxl_global_cnt[0] >= nVoxelX || vxl_global_cnt[1] >= nVoxelX) return;

		float u,v,w;
		if( cu_getBaryCoord2D(vxl_global_cnt, ftri + 0, ftri + 3, ftri + 6, u, v, w))
		{
			register int  new_z  = int(u * ftri[2] + v * ftri[5] + w * ftri[8]);
			register int  new_nZ = int(ftri[11] * cu_fNORMALFACTOR);//use the same normal vector.

			//write back data to global memory
			const int nSlot			= nVoxelX * nVoxelX;
			const int nSlotData = nSlot * CU_SLOT_CAPACITY_16;
			const int slot_ID		= (global_AABB_y0 + threadIdx.y) * nVoxelX + (global_AABB_x0 + threadIdx.x);//target slot ID.

			//memory index. do not change this order.
			CU_SLOT_BUFFER_TYPE*	memNPxl	= cu_nPixel + wrkID * nSlot		+ slot_ID;
			CU_SLOT_BUFFER_TYPE		n_pixel		= *memNPxl;//current number of pxls in the slot
			if(n_pixel < CU_SLOT_CAPACITY_16-1)
			{ 
				const CU_ULInt	slotdata_ID		= slot_ID * CU_SLOT_CAPACITY_16 + n_pixel;//pxl ID inside the target slot
				CU_ULInt	uliSlotDataIdx			= wrkID * nSlotData + slotdata_ID;
				CU_SLOT_BUFFER_TYPE*	memType	= cu_pType + uliSlotDataIdx;
				CU_SLOT_BUFFER_TYPE*	memZcrd	= cu_pZcrd + uliSlotDataIdx;
				CU_SLOT_BUFFER_TYPE*	memZnrm	= cu_pZnrm + uliSlotDataIdx;

				cu_Add(	 memNPxl, 1);//increase target slot's current # of pxls
				cu_Exch( memZcrd, new_z);
				cu_Exch( memZnrm, new_nZ);
				#ifdef _CUDA_USE_SPLIT_AL_BE_IN_VOXELIZE_STEP
				cu_Exch( memType, (new_nZ > 0) ? typeAl : typeBe);// = splitAlBe()
				#endif
			}
		}
		//end of per-THREAD operation ------------------------------
	}
	#else

	#endif


//**********************************************************************************************
//
// concurrent stream version
// 
// 
//**********************************************************************************************

	__global__ void cu_rotVoxel_Streamed_16x16(
		int nVoxelX, int nYPR, int nFlatTri, 
		int yprID, int triID_to_start,
		float* cu_m4x3,	float* cu_flattri0,	
		CU_SLOT_BUFFER_TYPE* cu_nPixel, CU_SLOT_BUFFER_TYPE* cu_pType, CU_SLOT_BUFFER_TYPE* cu_pZcrd, CU_SLOT_BUFFER_TYPE* cu_pZnrm)
	{
#if 0
		//Note: blockDim  = ( nTriToWork, 1, 1) x (16,16,1)
		int wrkID = blockIdx.x;//[nTriToWork]
		int triID	= triID_to_start + wrkID;//triangle ID to work in this stream.
		const int thID	= threadIdx.y * blockDim.x + threadIdx.x;//this block is (16,16,1) size.
#else
	//Note: blockDim  = ( nWorksPerBlocks, 1, 1) x (maxD,	maxD, CU_TRI_PER_WORK)
		int triID0	= triID_to_start + blockIdx.x * CU_TRI_PER_WORK;//global tri ID
		const int thID	= threadIdx.y * blockDim.x + threadIdx.x;
		const int thIDz = threadIdx.z;//local tri ID, [CU_TRI_PER_WORK]
#endif

		if (triID0 + thIDz>= nFlatTri ||yprID >= nYPR) return;

		__shared__ float ftri[CU_TRI_PER_WORK][CU_FLATTRI_SIZE_16], m4x3[CU_MATRIX_SIZE_12];

		//per-BLOCK operation +++++++++++++++
		if (thIDz==0 && thID <  CU_MATRIX_SIZE_12)	{//copy matrix data to shared memory.. 이건 모든 삼각형에 대해 공통.
				m4x3[thID]	= cu_m4x3[yprID * CU_MATRIX_SIZE_12 + thID];
		}		
		__syncthreads();


		if (thID < CU_FLATTRI_SIZE_16)	{//copy tri coord data to shared memory.각 삼각형별로 데이터 가져오기.
			ftri[thIDz][thID]	= cu_flattri0[(triID0 + thIDz) * CU_FLATTRI_SIZE_16 + thID];
		}
		__syncthreads();
		//matrix operation(YPR rotation + translation of AABB corner to origin)
		if(thID<=3) {			cu_matrixOp(m4x3, &ftri[thIDz][thID * 3]);		}
		__syncthreads();

		if(thID<=1) {//thID = 0, 1
			ftri[thIDz][12 + thID] = min(min(ftri[thIDz][0 + thID], ftri[thIDz][3+ thID]), ftri[thIDz][6 + thID]);//global_AABB_x0, _y0
		}
		else if(thID<=4) {//thID = 2,3,4
			ftri[thIDz][9+(thID-2)] -= m4x3[4*(thID-2)+3];//eliminate translation from normal vector
		}
		__syncthreads();


		//  로컬 픽셀좌표는 (threadIdx.x,threadIdx.y)
		//  global 좌표계는 (AABB_x0 + threadIdx.x, AABB_y0 + threadIdx.y)임.
		const int& global_AABB_x0 = ftri[thIDz][12];
		const int& global_AABB_y0 = ftri[thIDz][13];
		float vxl_global_crd[2] = { //global 좌표계에서의 현재 복셀의 위치.
						(global_AABB_x0 + threadIdx.x) + cu_HALF_VOXEL_SIZE,
						(global_AABB_y0 + threadIdx.y) + cu_HALF_VOXEL_SIZE };
		if (vxl_global_crd[0] >= nVoxelX || vxl_global_crd[1] >= nVoxelX) return;
		const int slot_ID		= (global_AABB_y0 + threadIdx.y) * nVoxelX + (global_AABB_x0 + threadIdx.x);//target slot ID.
		float u = -1.f,v = -1.f,w = -1.f;//barycentric coord.

		//(1) exceptional case: for very small triangle
		//if(1 && thID ==0 && ftri[thIDz][15]/*=ttri.AABB.GetMaxSpan()*/ < 1.1)
		//{
		//	u = v = w = 0.33333f;
		//	vxl_global_crd[0] = ftri[thIDz][0] * u + ftri[thIDz][3] * v + ftri[thIDz][6] * w;
		//	vxl_global_crd[1] = ftri[thIDz][2] * u + ftri[thIDz][4] * v + ftri[thIDz][7] * w;
		//}

		//(2) general case. use barycentric coordinate
		if(u < 0.f && !cu_getBaryCoord2D(vxl_global_crd, &ftri[thIDz][0], &ftri[thIDz][3], &ftri[thIDz][6], u, v, w))
		{
			return;
		}

		//write back data to global memory.do not change this order.
		CU_SLOT_BUFFER_TYPE*	memNPxl	= cu_nPixel + slot_ID;
		if( *memNPxl < CU_SLOT_CAPACITY_16-1)
		{ 
			const CU_ULInt	slotdata_ID		= slot_ID * CU_SLOT_CAPACITY_16 + *memNPxl;//pxl ID inside the target slot. 순서주의: cu_Add(	 memNPxl, 1) 앞에 와야 한다.

			cu_Add(	 memNPxl, 1);//increase target slot's current # of pxls

			int new_z  = int(u * ftri[thIDz][2] + v * ftri[thIDz][5] + w * ftri[thIDz][8]);
			int new_nZ = int(ftri[thIDz][11] * cu_fNORMALFACTOR);//use the same normal vector.

			cu_Exch( cu_pZcrd + slotdata_ID, new_z);
			cu_Exch( cu_pZnrm + slotdata_ID, new_nZ);
			#ifdef _CUDA_USE_SPLIT_AL_BE_IN_VOXELIZE_STEP
			cu_Exch( cu_pType + slotdata_ID, (new_nZ > 0) ? typeAl : typeBe);// = splitAlBe()
			#endif
		}

		//+++++++++++++++++++++++++++++++++++
	}
