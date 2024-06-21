// TomoNVC_Win32.cpp : Defines the exported functions for the DLL.

#include "pch.h"
#include "framework.h"
#include "Tomo_GPU2024.h"
#include "cpu_src\STomoPixel.h"
#include "cpu_src\STPSlot.h"
#include "cpu_src\STomoNV_TMPxl.h"
#include "cpu_src\STomoNV_INT3.h"
#include "cpu_src\STomoNV_CvxH.h"
#include "cpu_src\SMatrix4f.h"
#include "cuda_src\STomoNV_CUDA.cuh"

//#include "STomoNV_CvxH.h"
//#include "test_functions.hpp" //obsolete..

#include <algorithm>  // std::find_if
#include <thread>
#include <iostream>//std::cout


//global variables
FLOAT32** vtx = nullptr;
FLOAT32** nrm = nullptr;
INT16  ** tri = nullptr;
INT16 nV = 0;
INT16 nT = 0;
INT16 n_pxls = 0;
MESH_ELE_ID_TYPE*  nData2i = nullptr;
INT16** pData2i = nullptr;
FLOAT32* Mss = nullptr;
FLOAT32* Mo = nullptr;
FLOAT32* Vtc = nullptr;//debug. for p-orbital test
FLOAT32 optimal_YPR[3];
bool  g_bUseExplicitSS = false;
STomoVolMassInfo VolMassInfo;

TPVector al_pxls, be_pxls, TC_pxls, NVB_pxls, NVA_pxls, Vo_pxls, Vss_pxls, SS_pxls, SSB_pxls, SSA_pxls, Bed_pxls;

using namespace Tomo;

inline FLOAT32 toRadian(INT16 a) { return a * 3.141592 / 180.; }


void  getCVVoxels_INT3(TVVector& CV_vxls, const S3DPrinterInfo& _info, FLOAT32 _yaw, FLOAT32 _pitch, FLOAT32 _roll)
{
	STomoNV_INT3 tempNV;
	tempNV.printer_info = _info;
	tempNV.bStoreCVV = true;
	tempNV.printer_info.bUseClosedVolumeVoxel = false;
	S3DPrinterInfo& P_info = tempNV.printer_info;

	P_info.yaw = _yaw;  P_info.pitch = _pitch; P_info.roll = _roll;
	tempNV.Rotate();
	tempNV.Pixelize(tempNV.CV_vxls);
	tempNV.Pairing();

	moveVoxelCentersToOrigin(tempNV.CV_vxls);//thread_func()에서 회전시킬 거니까, 미리 원점으로 옮겨 둔다.
	rotateVoxels(tempNV.CV_vxls, -P_info.yaw, -P_info.pitch, -P_info.roll);
	moveVoxelCentersToOrigin(tempNV.CV_vxls);
	CV_vxls.insert(CV_vxls.end(), tempNV.CV_vxls.begin(), tempNV.CV_vxls.end()); //CVV_vxls << tempNV.CV_vxls; 너무 느림
}

void  getCVVoxels_TMPxl(TVVector& CV_vxls, const S3DPrinterInfo& _info, FLOAT32 _yaw, FLOAT32 _pitch, FLOAT32 _roll)
{
	STomoNV_TMPxl tempNV;
	tempNV.printer_info = _info;
	tempNV.printer_info.bUseClosedVolumeVoxel = false;
	S3DPrinterInfo& P_info = tempNV.printer_info;

	P_info.yaw = _yaw;  P_info.pitch = _pitch; P_info.roll = _roll;
	tempNV.Rotate();
	tempNV.Pixelize(tempNV.CV_vxls);
	//tempNV.Pairing();//여기서는 pairing 하지 말 것.
	tempNV.GetCVVoxelsFromSlots(tempNV.CV_vxls);

	moveVoxelCentersToOrigin(tempNV.CV_vxls);//thread_func()에서 회전시킬 거니까, 미리 원점으로 옮겨 둔다.
	rotateVoxels(tempNV.CV_vxls, -P_info.yaw, -P_info.pitch, -P_info.roll);
	moveVoxelCentersToOrigin(tempNV.CV_vxls);
	CV_vxls.insert(CV_vxls.end(), tempNV.CV_vxls.begin(), tempNV.CV_vxls.end()); //CVV_vxls << tempNV.CV_vxls; 너무 느림
}


INT16* pxlsToDat2i(TPVector& pxls, MESH_ELE_ID_TYPE& n_pxl)
{
	n_pxl = pxls.size();
	if(n_pxl<=0) return nullptr;

	INT16* _Data2i = new INT16[n_pxl * g_nPixelFormat];
	memset(_Data2i, 0x00, sizeof(INT16) * n_pxl * g_nPixelFormat);
	MESH_ELE_ID_TYPE p = 0;
	for (TPIterator pIt = pxls.begin(); pIt != pxls.end(); ++pIt, ++p)
	{
		pIt->DumpTo(_Data2i + p * g_nPixelFormat);
	}
	return _Data2i;
}

INT16*       getpData2i(Tomo::enumPixelType iSubPixel) { return ::pData2i[static_cast<int>(iSubPixel)];}
MESH_ELE_ID_TYPE  getnData2i(Tomo::enumPixelType iSubPixel) { return ::nData2i[static_cast<int>(iSubPixel)];}
FLOAT32* getMss(void) { return ::Mss;}
FLOAT32* getMo(void)  { return ::Mo; }
FLOAT32* getVtc(void) { return ::Vtc; }
FLOAT32* getVolMassInfo(void) 
{ 
	return ::VolMassInfo.dData; 
}

void  OnDestroy(void)
{
	if(pData2i!=nullptr)
	{
		for (int i = 0; i < static_cast<int>(enumPixelType::eptNumberOfSubPixels); i++)    {      if(pData2i[i] != nullptr) delete[] pData2i[i];    }
		delete[] pData2i;    pData2i = nullptr;
	}
	vtx = nullptr;  nrm = nullptr;  tri = nullptr;
	if (Mss != nullptr) { delete[] Mss;  Mss = nullptr;  }
	if (Mo != nullptr)  { delete[] Mo;   Mo = nullptr; }
	if (Vtc != nullptr) { delete[] Vtc;  Vtc = nullptr; }

	cudaDeviceReset();
}

MESH_ELE_ID_TYPE _find1stOptimal(MESH_ELE_ID_TYPE _nData, FLOAT32* _pData)
{
	MESH_ELE_ID_TYPE min_index = 0;
	FLOAT32 min_value = FLOAT32( 1e5);
	for (MESH_ELE_ID_TYPE i = 0; i < _nData; i++)  {  if (_pData[i] < min_value)  { min_value = _pData[i]; min_index = i;  }  }
	return min_index;
}

template <typename T> void thread_func(T* _pNV, int thread_id, FLOAT32* _YPR, int ypr_id, const TVVector& CV_vxls)
{//멀티스레드 적용을 위한 함수
	T* nv = _pNV + thread_id;

	S3DPrinterInfo& P_info = nv->printer_info;
	P_info.yaw = _YPR[ypr_id * 3 + 0];//YPR data is input here.
	P_info.pitch = _YPR[ypr_id * 3 + 1];
	P_info.roll = _YPR[ypr_id * 3 + 2];

	nv->Rotate();
	nv->Pixelize(CV_vxls);
	nv->Pairing();
	nv->GenerateBed();
	nv->Calculate();

	Mo[ypr_id] = nv->vm_info.Mo;
	Mss[ypr_id] = nv->vm_info.Mss;
	Vtc[ypr_id] = nv->vm_info.Vtc;
}

template <class T>
MESH_ELE_ID_TYPE  _TomoNV_Function_Call(
	FLOAT32* _float32_info_x12,
	MESH_ELE_ID_TYPE* _int32_info_x9,
	FLOAT32* _YPR,
	MESH_ELE_ID_TYPE* _tri,
	FLOAT32* _vtx,
	FLOAT32* _vtx_nrm,
	FLOAT32* _tri_nrm,
	MESH_ELE_ID_TYPE* _chull_tri,
	FLOAT32* _cvhull_vtx,
	FLOAT32* _chull_trinrm
)
{
	startTimer();

	S3DPrinterInfo info(_float32_info_x12, _int32_info_x9, _tri, _vtx, _vtx_nrm, _tri_nrm, _chull_tri, _cvhull_vtx, _chull_trinrm, 0, 0, 0);//takes some memory.

	Mss = new FLOAT32[info.nYPR + 2];
	Mo = new FLOAT32[info.nYPR + 2];
	Vtc = new FLOAT32[info.nYPR + 2];

	//mult-thread info.
	const auto processor_count = std::thread::hardware_concurrency();
	int nThread = min(processor_count - 1, info.nYPR);
	int nBlock = info.nYPR / nThread;
	int nBlRest = info.nYPR % nThread;


	T *pNV = new T[nThread +2];

	for (int thread_id = 0; thread_id < nThread; thread_id++)
	{
		pNV[thread_id].printer_info  = info;
	}

	//for CVV version
	TVVector CV_vxls;
	if(info.bUseClosedVolumeVoxel)
	{//복셀을 외부에서 미리 생성해 둔다.
		getCVVoxels_INT3(CV_vxls, info, 0, 0, 0);
		//getCVVoxels_INT3(CV_vxls, info, toRadian(90.), 0, 0);
		//getCVVoxels_INT3(CV_vxls, info, 0, toRadian(90.), 0);
	}


#ifdef _DEBUG
	//non-threaded version
	int ypr_id = 0;
	for (int b = 0; b < nBlock; b++)
	{
		for (int thread_id = 0; thread_id < nThread; thread_id++)
		{ 
			thread_func<T>(pNV, thread_id, _YPR, ypr_id++, CV_vxls); 
		}
		if (info.bVerbose && b % nBlock == 0) { std::cout << "Step" << ypr_id + 1 << "/" << info.nYPR << std::endl; }
	}

	{ for (int thread_id = 0; thread_id < nBlRest; thread_id++) { thread_func<T>(pNV, thread_id, _YPR, ypr_id++, CV_vxls); }  }
#else
	//std::thread version
	int ypr_id = 0;
	for (int b = 0; b < nBlock; b++) //Repeat # of CPU core * integer(nBlock) jobs
	{
		std::vector< std::thread> thVec;
		for (int thread_id = 0; thread_id < nThread; thread_id++) { thVec.push_back(std::thread(thread_func<T>, pNV, thread_id, _YPR, ypr_id++, CV_vxls)); }
		for (auto& th : thVec) { th.join(); }

		if (info.bVerbose && b % 100 == 0) { std::cout << "Step" << ypr_id + 1 << "/" << info.nYPR << std::endl; }
}

	{ std::vector< std::thread> thVec; //Do the rest jobs
	for (int thread_id = 0; thread_id < nBlRest; thread_id++) { thVec.push_back(std::thread(thread_func<T>, pNV, thread_id, _YPR, ypr_id++, CV_vxls)); }
	for (auto& th : thVec) { th.join(); }  }

#endif

	MESH_ELE_ID_TYPE  optID = (info.nYPR > 10) ? _find1stOptimal( info.nYPR, Mss) : 0;
	if( info.nYPR >1)  thread_func<T>(pNV, 0, _YPR, optID, CV_vxls);//Find information of optID again.

	//prepare rendering data for python
	int nPixelType = static_cast<int>(enumPixelType::eptNumberOfSubPixels);
	::nData2i = new MESH_ELE_ID_TYPE[nPixelType];
	::pData2i = new INT16*     [nPixelType];

	for (int i = 0; i < nPixelType; i++)
	{
		TPVector tmp_pxls = pNV[0].slotsToPxls(enumPixelType(i));
		pData2i[i] = pxlsToDat2i( tmp_pxls, nData2i[i]);
	}

	if(info.bVerbose)
	{//Find SS_pxls for rendering.
		TPVector tmp_pxls = pNV[0].GetSSPixels(info.bUseExplicitSS);
		int _SS = static_cast<int>(enumPixelType::eptSS);
		pData2i[_SS] = pxlsToDat2i(tmp_pxls, nData2i[_SS]);
	}

	VolMassInfo = pNV[0].vm_info; //final result

	delete[] pNV;
	endTimer("TomoNV C++ DLL calculation ");
	return optID;
}

//python interface functions
MESH_ELE_ID_TYPE  TomoNV_TMPxl(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm)
{
	return  _TomoNV_Function_Call<STomoNV_TMPxl>(_float32_info_x12, _int32_info_x9, _YPR, _tri, _vtx, _vtx_nrm, _tri_nrm, _chull_tri, _chull_vtx, _chull_trinrm);
}

MESH_ELE_ID_TYPE TomoNV_INT3(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm)
{
	return  _TomoNV_Function_Call<STomoNV_INT3>(_float32_info_x12, _int32_info_x9, _YPR, _tri, _vtx, _vtx_nrm, _tri_nrm, _chull_tri, _chull_vtx, _chull_trinrm);
}

MESH_ELE_ID_TYPE TomoNV_CvxH(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm)
{
	return  _TomoNV_Function_Call<STomoNV_CvxH>(_float32_info_x12, _int32_info_x9, _YPR, _tri, _vtx, _vtx_nrm, _tri_nrm, _chull_tri, _chull_vtx, _chull_trinrm);
}



#ifdef _USE_CUDA_FOR_TOMONV_0
//CUDA버전 테스트 1.1 subdivision
TOMO_FLOAT32 a0[3] = { 0, 10, 10 };
TOMO_FLOAT32 a1[3] = { 0, 0, 10 };
TOMO_FLOAT32 a2[3] = { 10, 0, 10 };
TOMO_FLOAT32 an[3] = { 0, 0, 1 };

STomoVoxel al_0(a0, an);
STomoVoxel al_1(a1, an);
STomoVoxel al_2(a2, an);

STomoTriangle tri0(al_0, al_1, al_2);
std::cout << "before" << std::endl;
tri0.Print();

TTriVector tri_vec;
TOMO_FLOAT32 threshold = 8;
triDivide(tri_vec, tri0, threshold);
std::cout << "after" << std::endl;
for (auto& t : tri_vec)
{
	t.Print();
}
int i = 0;

#endif


#ifdef _DEBUG
#include <iostream>//std::cout
void  saveSTL(int nTri, FLOAT32 *_flattri)
{
	if(nTri == 0 || _flattri == nullptr) return;

	std::cout << "solid \"_flattri\" \n";
#if 1
	for (int t = 0; t < nTri; t++)
	{
		FlatTriInfo* pFT = (FlatTriInfo*) (_flattri + t * nFlatTriInfoSize);

		std::cout << "  facet normal " << pFT->tri_nrm[0] << " " << pFT->tri_nrm[1] << " " << pFT->tri_nrm[2] << "\n";
		std::cout << "    outer loop\n";
		std::cout << "      vertex " << pFT->vtx0[0] << " " << pFT->vtx0[1] << " " << pFT->vtx0[2] << "\n";
		std::cout << "      vertex " << pFT->vtx1[0] << " " << pFT->vtx1[1] << " " << pFT->vtx1[2] << "\n";
		std::cout << "      vertex " << pFT->vtx2[0] << " " << pFT->vtx2[1] << " " << pFT->vtx2[2] << "\n";
		std::cout << "    endloop\n";
		std::cout << "  endfacet\n";
	}
#else
	for( int t = 0 ;t < nTri ; t++)
	{
		for( int i = 0 ; i < nFlatTriInfoSize ; i++)
		{
			std::cout << _flattri[ t * nFlatTriInfoSize + i] << " ";
		}
		std::cout << std::endl;
	}
#endif
	std::cout << "endsolid \"_flattri\" " << std::endl;
}
#endif


MESH_ELE_ID_TYPE  TomoNV_CUDA(FLOAT32* _float32_info_x12, MESH_ELE_ID_TYPE* _int32_info_x9, FLOAT32* _YPR, MESH_ELE_ID_TYPE* _tri, FLOAT32* _vtx, FLOAT32* _vtx_nrm, FLOAT32* _tri_nrm, MESH_ELE_ID_TYPE* _chull_tri, FLOAT32* _chull_vtx, FLOAT32* _chull_trinrm)
{
	STomoNV_CUDA  Cuda1;
	S3DPrinterInfo& P_info = Cuda1.printer_info;

	P_info.Set(_float32_info_x12, _int32_info_x9, _tri, _vtx, _vtx_nrm, _tri_nrm, _chull_tri, _chull_vtx, _chull_trinrm, 0, 0, 0);//input mesh data. takes some memory.

	bool bUseCVV = P_info.bUseClosedVolumeVoxel;
	MESH_ELE_ID_TYPE nYPR = P_info.nYPR;
	int   nCHullVtx = P_info.nCHull_Vtx;

	Mss = new FLOAT32[nYPR + 2];
	Mo  = new FLOAT32[nYPR + 2];
	Vtc = new FLOAT32[nYPR + 2];
	memset(Mss, 0x00, sizeof(FLOAT32) * nYPR);
	memset(Mo , 0x00, sizeof(FLOAT32) * nYPR);
	memset(Vtc, 0x00, sizeof(FLOAT32) * nYPR);

	//for CVV version
	TVVector CV_vxls;
	P_info.SetMaxTriDiameter();//do subdivision and make triangles smaller than CU_TRI_MAX_DIAMETER.
	if(bUseCVV)
	{
		getCVVoxels_TMPxl(CV_vxls, P_info, 0, 0, 0);	//복셀을 외부에서 미리 생성해 둔다.

		P_info.GetYPR4x3Matrix( _YPR, CV_vxls);//prepare rotation matrices
	}
	else{ 
		P_info.GetYPR4x3Matrix( _YPR, nCHullVtx, _chull_vtx);//prepare rotation matrices
	}

	Cuda1.bWriteBackPxlsForRendering = (nYPR == 1);
	Cuda1.Run(CV_vxls, Mo, Mss);//find optimals 

	if(!P_info.bVerbose) return 0;

	//prepare rendering the 1st optimal
	MESH_ELE_ID_TYPE  optID = (nYPR > 1) ? _find1stOptimal(nYPR, Mss) : 0;//ID to display
	if(nYPR > 1)
	{//prepare optID's pxls. for rendering.
		P_info.nYPR = 1;  P_info.bVerbose = false;
		memcpy( _YPR, _YPR + optID * 3 , sizeof(FLOAT32) * 3);
		Cuda1.bWriteBackPxlsForRendering = true;
		(bUseCVV) ?		P_info.GetYPR4x3Matrix( _YPR, CV_vxls) :	P_info.GetYPR4x3Matrix( _YPR, nCHullVtx, _chull_vtx);//prepare rotation matrices
		Cuda1.Run(CV_vxls, nullptr, nullptr);
	}

	//deliver pxl data to python
	VolMassInfo = Cuda1.vm_info;//for debug. deliver volume/mass data to python's Print_tabbed() function. 
	int nPixelType = static_cast<int>(enumPixelType::eptNumberOfSubPixels);
	nData2i = new MESH_ELE_ID_TYPE[nPixelType];  pData2i = new INT16 * [nPixelType];
	for (int i = 0; i < nPixelType; i++) { nData2i[i] = 0; pData2i[i] = nullptr; }

	FLOAT32 *rotated_floattri = nullptr; 
	rotated_floattri = Cuda1.ReadPxls(0, nData2i, pData2i);//time consuming. for Python rendering.
#ifdef _DEBUG
	//if(Cuda1.printer_info.nFlatTri< 100) saveSTL(Cuda1.printer_info.nFlatTri, rotated_floattri);
#endif
	if(rotated_floattri!=nullptr) delete[] rotated_floattri;

	return 0;
}


