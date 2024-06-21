#pragma once

#include "Tomo_types.h"
#include "cpu_src\STomoPixel.h"
#include "cpu_src\STomoVoxel.h"

using namespace Tomo;

template <class T> MESH_ELE_ID_TYPE  _TomoNV_Function_Call(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm);
template <typename T> void thread_func(T* _pNV, int thread_id, FLOAT32* _YPR, int ypr_id, const  TVVector& CVV_vxls);//멀티스레드 적용을 위한 함수

MESH_ELE_ID_TYPE _find1stOptimal(MESH_ELE_ID_TYPE _nData, FLOAT32* _pData);

#ifdef __cplusplus
extern "C"
{
#endif

	DLLEXPORT INT16* getpData2i(Tomo::enumPixelType iPixel);
	DLLEXPORT MESH_ELE_ID_TYPE  getnData2i(Tomo::enumPixelType iPixel);
	DLLEXPORT FLOAT32* getMss(void);
	DLLEXPORT FLOAT32* getMo(void);
	DLLEXPORT FLOAT32* getVtc(void);//for p-orbital. python version.
	DLLEXPORT FLOAT32* getVolMassInfo(void);

	DLLEXPORT INT16* pxlsToDat2i(TPVector& pxls, MESH_ELE_ID_TYPE& n_pxl);

	DLLEXPORT void  OnDestroy(void);

	//python interface functions
	DLLEXPORT MESH_ELE_ID_TYPE  TomoNV_TMPxl(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm);
	DLLEXPORT MESH_ELE_ID_TYPE  TomoNV_INT3(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm);
	DLLEXPORT MESH_ELE_ID_TYPE  TomoNV_CvxH(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm);

	DLLEXPORT MESH_ELE_ID_TYPE  TomoNV_CUDA(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm);

#ifdef __cplusplus
}
#endif
