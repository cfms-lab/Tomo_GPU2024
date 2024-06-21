#include "STomoNV_CUDA0.cuh"
//#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#include "voxelize.cuh"
#include "slotPairing.cuh"
#include "atomicWrite.cuh"
#include "reducedSum.cuh"

using namespace Tomo;


__host__ STomoNV_CUDA0::STomoNV_CUDA0() : STomoNV_Base()
{
}

__host__ STomoNV_CUDA0::~STomoNV_CUDA0()
{
	Reset();
}

__host__ STomoNV_CUDA0::STomoNV_CUDA0(const STomoNV_CUDA0& Source)
{
	Init();
	_Copy(Source);
}

__host__ void	STomoNV_CUDA0::operator=(const STomoNV_CUDA0& Source)
{
	Reset();
	_Copy(Source);
}

__host__ void	STomoNV_CUDA0::_Copy(const STomoNV_CUDA0& Source)
{
	STomoNV_Base::_Copy(Source);
	memcpy(iData, Source.iData, siData);//params.
	memcpy(cu_fpData, Source.cu_fpData, cu_sfpData);//GPU mem

}

__host__ void	STomoNV_CUDA0::Reset(void)
{
	STomoNV_Base::Reset();
}

__host__ void	STomoNV_CUDA0::Init(void)
{
	STomoNV_Base::Init();
	memset(iData, 0x00, siData);//params.
	memset(cu_fpData, 0x00, cu_sfpData);//GPU mem

	nStream = 1;
}

__host__ void  STomoNV_CUDA0::initCudaMem(void)
{
	if(printer_info.bVerbose) cu_startTimer();

	cudaError_t cudaStatus = cudaSetDevice(0); cudaCheckError();

	unsigned long int uliSlotSize = s_nSlot * nWorksInBatch;
	unsigned long int uliSlotDataSize = s_nSlotData * nWorksInBatch;

	//cpu memroy - as 1D, pinned
	cpu_SlotData.Reset();
	cpu_SlotData.bDevice = false;
	cpu_SlotData.Malloc(nSlot);

	//gpu memory
	gpu_SlotDataVec.clear();
	for( int s = 0 ; s < nWorksInBatch ; s++)
	{
		SlotData sd;		
		sd.Malloc( nSlot);
		gpu_SlotDataVec.push_back(sd);
	}


#ifndef _CUDA_USE_ROTATE_AND_PIXELIZE_IN_ONE_STEP
	cudaMalloc((void**)&cu_FlatTri1, s_nFlatTri * nYPRInBatch);		cudaCheckError();
	cudaMemsetAsync(cu_FlatTri1, 0x00, s_nFlatTri * nYPRInBatch);	cudaCheckError();//not necessary. for debug
#endif

#ifndef _CUDA_USE_MULTI_STREAM
	cudaMalloc((void**)&cu_FlatTri0,	s_nFlatTri);			cudaCheckError();
	cudaMalloc((void**)&cu_m4x3,			s_m4x3 * nYPR);					cudaCheckError();
	cudaMemcpy(cu_FlatTri0,	printer_info.pFlatTri,	s_nFlatTri,	cudaMemcpyHostToDevice); cudaCheckError();//Time Consuming!!
	cudaMemcpy(cu_m4x3,			printer_info.YPR_m4x3,	s_m4x3 * nYPR,			cudaMemcpyHostToDevice); cudaCheckError();
#endif

	if(printer_info.bVerbose) cu_endTimer("CUDA init() ");
}

__host__ void  STomoNV_CUDA0::clearCudaMem(void)
{
	for( auto& sd : gpu_SlotDataVec)	{		sd.Free( );	}
	cpu_SlotData.Free();

	cudaFree((void*)cu_FlatTri0);
	cudaFree((void*)cu_m4x3);

#ifndef _CUDA_USE_ROTATE_AND_PIXELIZE_IN_ONE_STEP
	cudaFree(cu_FlatTri1);
#endif
	
}

__host__ void	STomoNV_CUDA0::Init_CUDA(void)
{
	clearCudaMem();

	nFlatTri	= printer_info.nFlatTri; 
	nYPR			= printer_info.nYPR;
	nVoxelY		= nVoxelX = printer_info.nVoxel;
	nSlot			= nVoxelX * nVoxelY;
	nSlotData	= nSlot * CU_SLOT_CAPACITY_16;

	s_nSlot			= sizeof(CU_SLOT_BUFFER_TYPE) * nSlot;
	s_nSlotData = sizeof(CU_SLOT_BUFFER_TYPE) * nSlotData;
	s_nFlatTri	= sizeof(float) * nFlatTri * nFlatTriInfoSize;
	s_m4x3			= sizeof(float) * CU_MATRIX_SIZE_12;// * nYPR;

	getBatchNumber();
	initCudaMem();
}


#include "../cpu_src/STomoPixel.h"
__host__ FLOAT32* STomoNV_CUDA0::ReadPxls(int wrkID, int* _nData2i, INT16** _pData2i)
{

	unsigned long uliSlotIdx = wrkID * s_nSlot;
	unsigned long uliSlotDataIdx = wrkID * s_nSlotData;

	gpu_SlotDataVec.at(wrkID).ReadPxls( 
		cpu_SlotData.cu_sNPxl, 
		cpu_SlotData.cu_sdType, 
		cpu_SlotData.cu_sdZcrd, 
		cpu_SlotData.cu_sdZnrm);

	//for rendering(time-consuming)
	TPVector pxls[(int)enumPixelType::espNumberOfSubPixels];
	vm_info.Va = vm_info.Vb = vm_info.Vss = 0;
	for (int slot_id = 0; slot_id < nSlot; slot_id++)
	{
		int p_x = slot_id % (nVoxelX);
		int p_y = slot_id / (nVoxelX);
		int S_L = cpu_SlotData.cu_sNPxl[slot_id];

		int ss_start_z = -1, ss_end_z = -1;//position of support structure segment
		for (int slotdata_id = 0; slotdata_id < S_L; slotdata_id++)
		{
			int p_z			= cpu_SlotData.cu_sdZcrd[slot_id * CU_SLOT_CAPACITY_16 + slotdata_id];
			int p_nZ		= cpu_SlotData.cu_sdZnrm[slot_id * CU_SLOT_CAPACITY_16 + slotdata_id];
			int p_type	= cpu_SlotData.cu_sdType[slot_id * CU_SLOT_CAPACITY_16 + slotdata_id];
			STomoPixel new_pxl(p_x, p_y, p_z, 0, 0, p_nZ, p_type);
			if (p_type & typeAl)
			{
				pxls[(int)enumPixelType::espAl].push_back(new_pxl);
				vm_info.Va += p_z;
			}
			if (p_type & typeSSA)
			{
				pxls[(int)enumPixelType::espSSA].push_back(new_pxl);
				vm_info.Vss -= p_z; ss_end_z = p_z; 
			}
			if (p_type & typeBe)
			{
				pxls[(int)enumPixelType::espBe].push_back(new_pxl);
				vm_info.Vb += p_z;
			}
			if (p_type & typeSSB)
			{
				pxls[(int)enumPixelType::espSSB].push_back(new_pxl);
				vm_info.Vss += p_z; ss_start_z = p_z;
			}

#if 1
			if(ss_start_z > -1 && ss_end_z > -1 && ss_start_z > ss_end_z)//느리고, 버그 있음.
			{
				for(int z = ss_start_z ; z >= ss_end_z ; z--)
				{
					STomoPixel ss_pxl(p_x, p_y, z, typeSS, 0, 0);
					pxls[(int)enumPixelType::espSS].push_back(ss_pxl);
				}
				ss_start_z = -1; ss_end_z = -1;
			}
#endif
		}

	}

#ifdef _DEBUG
	if(1)
	{
		int _n_Va = pxls[(int)enumPixelType::espAl].size();
		int _n_Vb = pxls[(int)enumPixelType::espBe].size();
	}
#endif

	if(_nData2i==nullptr) return nullptr;

	for (size_t pt = 0; pt < (int)enumPixelType::espNumberOfSubPixels; pt++)
	{
		size_t n_pxl = pxls[pt].size();
		_nData2i[pt] = n_pxl;
		_pData2i[pt] = new INT16[n_pxl * 6 + 2];
		for (int p = 0; p < n_pxl; p++)
		{
			pxls[pt].at(p).DumpTo(_pData2i[pt] + p * 6);
		}
	}

	//debug - cehck YPR rotation matrix 
	FLOAT32* rotated_pFlatTri = nullptr;
	return rotated_pFlatTri;
}



__host__ void STomoNV_CUDA0::RotateAndPixelize(int _nWorksInBatch, int yprID_to_start)
{
	//reset mem
	unsigned long int uliSlotSize			= s_nSlot * _nWorksInBatch;
	unsigned long int uliSlotDataSize	= s_nSlotData * _nWorksInBatch;

	for( auto& sd: gpu_SlotDataVec){ sd.SetZero();}

	const dim3 dgRot(nFlatTri, _nWorksInBatch);

	const dim3 dbRot(CU_BLOCK_SIZE_16, CU_BLOCK_SIZE_16);
	cu_rotVoxel_16x16 << < dgRot, dbRot>> > ( nFlatTri, nYPR, nVoxelX, yprID_to_start,
				cu_m4x3, cu_FlatTri0,		
				gpu_SlotDataVec.begin()->cu_sNPxl, gpu_SlotDataVec.begin()->cu_sdType, gpu_SlotDataVec.begin()->cu_sdZcrd, gpu_SlotDataVec.begin()->cu_sdZnrm);
	cudaCheckError();

}

__host__ void  STomoNV_CUDA0::Pairing(int _nWorksInBatch)
{
	int sin_theta_c_x1000 = - sin(printer_info.theta_c) * 1000.;//SS critical angle condition

	const dim3 dgSort(nSlot/*=256x256=65536*/, _nWorksInBatch);
	const dim3 dbSort(CU_SLOT_CAPACITY_16);
	ShouldSwap<CU_SLOT_BUFFER_TYPE> descendingOrder(true);
	cu_slotPairing<CU_SLOT_BUFFER_TYPE> << < dgSort, dbSort>> > (
				gpu_SlotDataVec.begin()->cu_sNPxl, 
				gpu_SlotDataVec.begin()->cu_sdType, 
				gpu_SlotDataVec.begin()->cu_sdZcrd, 
				gpu_SlotDataVec.begin()->cu_sdZnrm, 
				gpu_SlotDataVec.begin()->cu_sVo, 
				gpu_SlotDataVec.begin()->cu_sVss, 
				descendingOrder, nSlot, sin_theta_c_x1000);				
}

__host__ void	STomoNV_CUDA0::getBatchNumber(void)
{
	//get max. number of YPR batch based on avaiable GPU memory size.
	cudaMemGetInfo(&gpu_available_mem, &gpu_total_mem); cudaCheckError();
	float gpu_MB = gpu_available_mem / float(1<<20);//debug
	if(printer_info.bVerbose)	{		std::cout << "GPU available memory= " << int(gpu_MB) <<" [MB]" << std::endl;	}

	if (nYPR == 1) { nWorksInBatch = 1; return; }

	size_t mem_per_work = 0;
	mem_per_work += s_nSlot			* 3;//cu_sVo, cu_sVss, cu_sNPxl, 
	mem_per_work += s_nSlotData	* 3;//cu_sdType, cu_sdZcrd, cu_sdZnrm
#ifndef _CUDA_USE_ROTATE_AND_PIXELIZE_IN_ONE_STEP
	mem_per_work += s_nFlatTri		* 1;//cu_flatTri_YPR.  (cu_flatTri0는 하나만 있으니까 제외)
#endif
#ifdef _CUDA_USE_MULTI_STREAM
	gpu_available_mem -= s_nFlatTri * nStream;
#endif

	nWorksInBatch = int(gpu_available_mem * 0.80 / mem_per_work);//use 80% for saftety. 

	float mem_per_work_MB = mem_per_work / float(1<<20);//debug
	if(printer_info.bVerbose)	{		std::cout << "memory needed per work=" << mem_per_work_MB<< " [MB]" << std::endl;	}
}

__host__ void	STomoNV_CUDA0::ReducedSum(int _nWorksInBatch, float *_Mo, float *_Mss)
{
	if(_Mss==nullptr) return;

	for( int wrkID = 0 ; wrkID < _nWorksInBatch ; wrkID++)
	{
		vm_info.Vo	= reducedSum(nSlot, (int*) gpu_SlotDataVec.at(wrkID).cu_sVo);
		vm_info.Vss = reducedSum(nSlot,	(int*) gpu_SlotDataVec.at(wrkID).cu_sVss);
		vm_info.VolToMass(printer_info);
		*(_Mo + wrkID) = vm_info.Mo;
		*(_Mss + wrkID) = vm_info.Mss;
	}
}


__host__ void	STomoNV_CUDA0::Run(float *_Mo, float *_Mss)
{
	Init_CUDA();

	cu_startTimer();

	int yprID_to_start = 0;
	int nBatch		= (nYPR == 1)? 0 : nYPR / nWorksInBatch;
	int nWorsRest	= (nYPR == 1)? 1 : nYPR % nWorksInBatch;

	if(printer_info.bVerbose)	{		std::cout << "nYPRInBatch= " << nWorksInBatch <<" , nBatch=" << nBatch<< std::endl;	}

	//iterate nBatch times
	for (int b = 0; b < nBatch; b++)
	{
		RotateAndPixelize(		nWorksInBatch, yprID_to_start);
		Pairing(	nWorksInBatch);
		ReducedSum(nWorksInBatch, _Mo + yprID_to_start, _Mss + yprID_to_start);
		yprID_to_start += nWorksInBatch;
	}

	//the rest iteration
	RotateAndPixelize( nWorsRest, yprID_to_start);
	Pairing(	nWorsRest);
	ReducedSum(nWorsRest,	 _Mo + yprID_to_start, _Mss + yprID_to_start);
	yprID_to_start += nWorsRest;

	cudaDeviceSynchronize(); cudaCheckError();

	cu_endTimer("CUDA main loop: ");

}
