#include "STomoNV_CUDA.cuh"
//#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#include "step1_voxelize.cuh"
#include "step1_rotCVVoxel.cuh"
#include "step2_slotPairing.cuh"
#include "step3_generateBed.cuh"
#include "step4_reducedSum.cuh"
#include "atomicWrite.cuh"

using namespace Tomo;
ShouldSwap<CU_SLOT_BUFFER_TYPE> descendingOrder(true);

__host__ STomoNV_CUDA::STomoNV_CUDA() : STomoNV_Base()
{
}

__host__ STomoNV_CUDA::~STomoNV_CUDA()
{
	Reset();
}

__host__ STomoNV_CUDA::STomoNV_CUDA(const STomoNV_CUDA& Source)
{
	Init();
	_Copy(Source);
}

__host__ void	STomoNV_CUDA::operator=(const STomoNV_CUDA& Source)
{
	Reset();
	_Copy(Source);
}

__host__ void	STomoNV_CUDA::_Copy(const STomoNV_CUDA& Source)
{
	STomoNV_Base::_Copy(Source);
	memcpy(iData, Source.iData, siData);//params.
	memcpy(cu_fpData, Source.cu_fpData, cu_sfpData);//GPU mem
	bWriteBackPxlsForRendering = Source.bWriteBackPxlsForRendering;
}

__host__ void	STomoNV_CUDA::Reset(void)
{
	STomoNV_Base::Reset();
}

__host__ void	STomoNV_CUDA::Init(void)
{
	STomoNV_Base::Init();
	memset(iData, 0x00, siData);//params.
	memset(cu_fpData, 0x00, cu_sfpData);//GPU mem

	nStream = 1;
	bWriteBackPxlsForRendering = true;
}

__host__ void  STomoNV_CUDA::initCudaMem(void)
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

__host__ void  STomoNV_CUDA::clearCudaMem(void)
{
	for( auto& sd : gpu_SlotDataVec)	{		sd.Free( );	}
	cpu_SlotData.Free();

	cudaFree((void*)cu_FlatTri0);
	cudaFree((void*)cu_m4x3);
	cudaFree((void*)cu_CVVoxels);
	cudaFreeHost((void*)ho_CVVoxels);

#ifndef _CUDA_USE_ROTATE_AND_PIXELIZE_IN_ONE_STEP
	cudaFree(cu_FlatTri1);
#endif
	
}

#include "../cpu_src/STomoPixel.h"
__host__ FLOAT32* STomoNV_CUDA::ReadPxls(int wrkID, int* _nData2i, INT16** _pData2i)
{

	unsigned long uliSlotIdx = wrkID * s_nSlot;
	unsigned long uliSlotDataIdx = wrkID * s_nSlotData;

	gpu_SlotDataVec.at(wrkID).ReadPxls( 
		cpu_SlotData.cu_sNPxl, 
		cpu_SlotData.cu_sdType, 
		cpu_SlotData.cu_sdZcrd, 
		cpu_SlotData.cu_sdZnrm);

	//for rendering(time-consuming)
	TPVector pxls[(int)enumPixelType::eptNumberOfSubPixels];
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
				pxls[(int)enumPixelType::eptAl].push_back(new_pxl);
				vm_info.Va += p_z;
			}
			if (p_type & typeSSA)
			{
				pxls[(int)enumPixelType::eptSSA].push_back(new_pxl);
				vm_info.Vss -= p_z; ss_end_z = p_z; 
			}
			if (p_type & typeBe)
			{
				pxls[(int)enumPixelType::eptBe].push_back(new_pxl);
				vm_info.Vb += p_z;
			}
			if (p_type & typeSSB)
			{
				pxls[(int)enumPixelType::eptSSB].push_back(new_pxl);
				vm_info.Vss += p_z; ss_start_z = p_z;
			}
			if (p_type & typeBed)
			{
				pxls[(int)enumPixelType::eptBed].push_back(new_pxl);
				vm_info.Vbed += p_z;
			}

#if 1
			if(ss_start_z > -1 && ss_end_z > -1 && ss_start_z > ss_end_z)//느리고, 버그 있음.
			{
				for(int z = ss_start_z ; z >= ss_end_z ; z--)
				{
					STomoPixel ss_pxl(p_x, p_y, z, typeSS, 0, 0);
					pxls[(int)enumPixelType::eptSS].push_back(ss_pxl);
				}
				ss_start_z = -1; ss_end_z = -1;
			}
#endif
		}

	}

#ifdef _DEBUG
	if(1)
	{
		int _n_Va = pxls[(int)enumPixelType::eptAl].size();
		int _n_Vb = pxls[(int)enumPixelType::eptBe].size();
	}
#endif

	if(_nData2i==nullptr) return nullptr;

	for (size_t pt = 0; pt < (int)enumPixelType::eptNumberOfSubPixels; pt++)
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




//**********************************************************************************************
//
//	concurrent stream version
// 
//**********************************************************************************************
__host__ void	STomoNV_CUDA::getBatchNumber(void)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0);
	if(!prop.deviceOverlap)
	{
		std::cout<< "ERROR: deviceOverlap not supported" << std::endl;
	}
	
	if(printer_info.bVerbose) std::cout<< "CUDA compute capability=" << prop.major << "."<< prop.minor << std::endl;

	nMultiProcessor = prop.multiProcessorCount;
	nMaxThreadPerMP = prop.maxThreadsPerMultiProcessor;
	nMaxThreads			= nMultiProcessor * nMaxThreadPerMP;
	nMaxConcurrentStream = 8;//ToDo: API?

	//nStream = nMultiProcessor;//<= nMaxConcurrentStream
	if(nYPR >1) 	{	nWorksInBatch = nStream = nMultiProcessor;}//(size_t) ::min( int(nMultiProcessor),8); }
	else	{	nWorksInBatch = nStream = 1;}
}

__host__ void	STomoNV_CUDA::Init_CUDA(void)
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
	s_m4x3			= sizeof(float) * CU_MATRIX_SIZE_12;

	getBatchNumber();
	initCudaMem();
}

#ifdef _CUDA_USE_SERIALIZED_VO_VSS_MEMORY
void	 STomoNV_CUDA::__Malloc_Serialized_Vo_Vss_memory(void)
{
	for(int s = 0 ; s < nStream ; s++)
	{
		SlotDataIterator sdIt = gpu_SlotDataVec.begin() + s;
		cudaFree(sdIt->cu_sVo);
		cudaFree(sdIt->cu_sVss);
	}

	CU_SLOT_BUFFER_TYPE *cu_buf = 0;
	cudaMalloc( (void**)&cu_buf,				sizeof(CU_SLOT_BUFFER_TYPE) * nSlot * 2* nStream);	cudaCheckError();
	for(int s = 0 ; s < nStream ; s++)
	{
		SlotDataIterator sdIt = gpu_SlotDataVec.begin() + s;
		sdIt->cu_sVo  = cu_buf + nSlot * (2* s + 0);
		sdIt->cu_sVss = cu_buf + nSlot * (2* s + 1);
	}
}

void	 STomoNV_CUDA::__Free_Serialized_Vo_Vss_memory(void)
{
	cudaFree( gpu_SlotDataVec.begin()->cu_sVo);
	for(int s = 0 ; s < nStream ; s++)
	{
		SlotDataIterator sdIt = gpu_SlotDataVec.begin() + s;
		sdIt->cu_sVo	= nullptr;
		sdIt->cu_sVss =  nullptr;
	}
}
#endif

__host__ void	STomoNV_CUDA::Run(const TVVector& CV_vxls, float *_Mo, float *_Mss)
{
	Init_CUDA();	

	if(printer_info.bVerbose)	{		std::cout << "nStream= " << nStream <<" , nWorksInBatch=" << nWorksInBatch<< std::endl;	}

	//send tri data to gpu
	unsigned long int uliSlotSize			= s_nSlot * nStream;
	unsigned long int uliSlotDataSize = s_nSlotData * nStream;
	cudaMalloc((void**)&cu_FlatTri0,		s_nFlatTri	* nStream);	cudaCheckError();
	cudaMalloc((void**)&cu_m4x3,				s_m4x3*nYPR	* nStream);	cudaCheckError();

	cudaMalloc((void**)&cu_sum_buffer,	sizeof(int)* 2	* nStream);	cudaCheckError();//Step4_ReducedSum_Batch()를 사용할 경우를 위해 넉넉하게 잡는다.
	cudaMallocHost((void**)&sum_buffer,	sizeof(int)* 2	* nStream);	cudaCheckError();
	
#ifdef _CUDA_USE_SERIALIZED_VO_VSS_MEMORY
if(nYPR>1) __Malloc_Serialized_Vo_Vss_memory();
#endif

	cudaMemcpy(cu_FlatTri0,	printer_info.pFlatTri,	s_nFlatTri,			cudaMemcpyHostToDevice); cudaCheckError();
	cudaMemcpy(cu_m4x3,			printer_info.YPR_m4x3,	s_m4x3 * nYPR,	cudaMemcpyHostToDevice); cudaCheckError();
	if(printer_info.bUseClosedVolumeVoxel)	{ PrepareCVVoxels( CV_vxls);	}

	if(printer_info.bVerbose) cu_startTimer();

		for( auto& sd: gpu_SlotDataVec) 	{ 			sd.CreateStream(); }//cuda 스트림 생성 & 이벤트 등록.

	cudaEvent_t last_event;
	for( int yprID = 0 ; yprID < nYPR ; yprID += nStream)
	{
		for(int s = 0 ; s < nStream ; s++)		
		{
			if(yprID + s < nYPR)			
			{
				SlotDataIterator sdIt = gpu_SlotDataVec.begin() + s;
				Step1_RotateAndPixelize( sdIt, yprID + s); 
				Step2_Pairing(sdIt);
				Step3_GenerateBed(sdIt);
#ifdef _CUDA_USE_REDUCED_SUM_BATCH
				cudaEventRecord( sdIt->event, sdIt->stream);	
#else		
				Step4_SlotSum( sdIt, cu_sum_buffer + 2 * s, sum_buffer + 2 * s, 	_Mo + yprID + s, _Mss + yprID + s);
#endif
			}
		}
#ifdef _CUDA_USE_REDUCED_SUM_BATCH
		Step4_SlotSum_Batch(yprID, _Mo, _Mss);//step1~3에서 계산한 Mo, Mss값을 nStream개 한꺼번에 합산한다. 
#endif
	}
	cudaDeviceSynchronize(); cudaCheckError();
	if (printer_info.bVerbose) cu_endTimer("CUDA main loop: ");

		for( auto& sd: gpu_SlotDataVec) 	{ 			sd.DestroyStream(); }

#ifdef _CUDA_USE_SERIALIZED_VO_VSS_MEMORY
if(nYPR>1) __Free_Serialized_Vo_Vss_memory();
#endif
}


__host__ void STomoNV_CUDA::Step1_RotateAndPixelize( SlotDataIterator sdIt, int yprID_to_start)
{
	if(cu_CVVoxels!=nullptr)  return Step1_RotateCVVoxels( sdIt, yprID_to_start);//CVVoxel을 재활용해서 쓰는 경우.

	//이 버전은 ypr을 _nStream개 처리한다.
	int problem_size = nFlatTri;
	int blocksize = CU_TRI_PER_WORK;
	int gridsize  = (problem_size + blocksize -1) / blocksize;
	const dim3 dgStep1( gridsize);//=nWorksPerBlocks
	const dim3 dbStep1(	printer_info.TriMaxDiameter,	printer_info.TriMaxDiameter, blocksize);//각 삼각형을 TriMaxDiameter x TriMaxDiameter 크기의 격자에 넣고 복셀화한다. 

	int triID_to_start = 0;//obsolete..

	sdIt->SetZero();
	cu_rotVoxel_Streamed_16x16 << < dgStep1, dbStep1, 0, sdIt->stream >> > ( 
					nVoxelX, nYPR, nFlatTri,	//constants
					yprID_to_start,			//variables
					triID_to_start, 
					cu_m4x3					+ 0 * CU_MATRIX_SIZE_12, //input data
					cu_FlatTri0			+ 0 * nFlatTri,		
					sdIt->cu_sNPxl,
					sdIt->cu_sdType,
					sdIt->cu_sdZcrd,
					sdIt->cu_sdZnrm	);
}

__host__ void  STomoNV_CUDA::Step2_Pairing(SlotDataIterator sdIt)
{
	int sin_theta_c_x1000 = - sin(printer_info.theta_c) * 1000.;//SS critical angle condition

	int problem_size = nSlot;
	int blocksize = CU_SLOTS_PER_WORK;
	int gridsize  = (problem_size + blocksize -1) / blocksize;
	const dim3 dgStep2( gridsize);
	const dim3 dbStep2( CU_SLOT_CAPACITY_16, blocksize);

	cu_slotPairing_Streamed<CU_SLOT_BUFFER_TYPE> << < dgStep2, dbStep2, 0, sdIt->stream>> > (
				nSlot, descendingOrder, sin_theta_c_x1000,//constants
				bWriteBackPxlsForRendering,//variables
				sdIt->cu_sNPxl, //input & output
				sdIt->cu_sVo, 
				sdIt->cu_sVss,
				sdIt->cu_sdType, 
				sdIt->cu_sdZcrd,
				sdIt->cu_sdZnrm);cudaCheckError();
}


__host__ void	STomoNV_CUDA::Step4_SlotSum(SlotDataIterator sdIt, int *_cu_reduced_sum_buffer, int *_reduced_sum_buffer, float *_Mo , float *_Mss)
{
	cudaMemsetAsync(_cu_reduced_sum_buffer, 0, sizeof(int) * 2 * 1, sdIt->stream);
	reducedSum_Streamed(nSlot, (int*)sdIt->cu_sVo, sdIt->stream, _cu_reduced_sum_buffer);
	reducedSum_Streamed(nSlot, (int*)sdIt->cu_sVss, sdIt->stream, _cu_reduced_sum_buffer + 1);
	cudaMemcpy(_reduced_sum_buffer, _cu_reduced_sum_buffer, sizeof(int) * 2, cudaMemcpyDeviceToHost); cudaCheckError();

	vm_info.Vo = *(_reduced_sum_buffer + 0);
	vm_info.Vss = *(_reduced_sum_buffer + 1);
	vm_info.VolToMass(printer_info);
	if(_Mo !=nullptr) *(_Mo) = vm_info.Mo;
	if(_Mss != nullptr) *(_Mss) = vm_info.Mss;
}

__host__ void	STomoNV_CUDA::Step4_SlotSum_Batch(int yprID, float *_Mo, float *_Mss) 
{
	if(_Mss==nullptr) return;

#if defined( _CUDA_USE_SERIALIZED_VO_VSS_MEMORY)
cudaMemset( cu_reduced_sum_buffer, 0, sizeof(int) * 2 * nStream);
reducedSum_2d( nStream * 2, nSlot, gpu_SlotDataVec.begin()->cu_sVo, cu_reduced_sum_buffer);
cudaMemcpy( reduced_sum_buffer,  cu_reduced_sum_buffer, sizeof(int) * 2 * nStream, cudaMemcpyDeviceToHost); cudaCheckError();

		for(int s = 0 ; s < nStream ; s++)
		{
			if(yprID + s < nYPR)
			{
				vm_info.Vo  = reduced_sum_buffer[2 * s];
				vm_info.Vss = reduced_sum_buffer[2 * s + 1];
				vm_info.VolToMass(printer_info);
				*(_Mo  + yprID + s) = vm_info.Mo;
				*(_Mss + yprID + s) = vm_info.Mss;
			}
		}

#elif defined( _CUDA_USE_MULTI_STREAM)
		//여기는 별도로 Vo_stream, Vss_stream을 사용한다. 
		for(int s = 0 ; s < nStream ; s++)
		{
			if(yprID + s < nYPR)
			{
				SlotDataIterator sdIt = gpu_SlotDataVec.begin() + s;
#ifdef _CUDA_USE_REDUCED_SUM_BATCH
				cudaStreamWaitEvent(sdIt->Vo_stream, sdIt->event);
				cudaStreamWaitEvent(sdIt->Vss_stream, sdIt->event);
				if(s==0) cudaMemsetAsync( cu_sum_buffer, 0, sizeof(int) * 2 * nStream, sdIt->Vo_stream);
				reducedSum_Streamed(nSlot, (int*) sdIt->cu_sVo , sdIt->Vo_stream, cu_sum_buffer + 2 * s);
				reducedSum_Streamed(nSlot, (int*) sdIt->cu_sVss, sdIt->Vss_stream, cu_sum_buffer + 2 * s + 1);
#endif
			}
		}
		cudaMemcpy( sum_buffer,  cu_sum_buffer, sizeof(int) * 2 * nStream, cudaMemcpyDeviceToHost); cudaCheckError();

		for(int s = 0 ; s < nStream ; s++)
		{
			if(yprID + s < nYPR)
			{
				vm_info.Vo  = sum_buffer[2 * s];
				vm_info.Vss = sum_buffer[2 * s + 1];
				vm_info.VolToMass(printer_info);
				*(_Mo  + yprID + s) = vm_info.Mo;
				*(_Mss + yprID + s) = vm_info.Mss;
			}
		}

#else
		for(int s = 0 ; s < nStream ; s++)
		{
			if(yprID + s < nYPR)
			{
				vm_info.Vo  = reducedSum(nSlot, (int*) gpu_SlotDataVec.at(s).cu_sVo);
				vm_info.Vss = reducedSum(nSlot,	(int*) gpu_SlotDataVec.at(s).cu_sVss);
				vm_info.VolToMass(printer_info);
				*(_Mo  + yprID + s) = vm_info.Mo;
				*(_Mss + yprID + s) = vm_info.Mss;
			}
		}

#endif
}

//----------------------------------------------------------------------------------------------
//
//	CVVoxels version
// 
//----------------------------------------------------------------------------------------------

__host__ void  STomoNV_CUDA::PrepareCVVoxels(const TVVector& CV_vxls)
{
	nCVVoxel=CV_vxls.size();	if(nCVVoxel<=0) return; 
	
	size_t s_CVVoxels = nCVVoxel * sizeof(float) * 6;
	cudaMallocHost( &ho_CVVoxels, s_CVVoxels);//(x,y,z, nX, nY, nZ)
	cudaMalloc( &cu_CVVoxels, s_CVVoxels);
	for(int p = 0 ; p < nCVVoxel ; p++)	{		(CV_vxls.begin() + p)->DumpTo( ho_CVVoxels + p * 6);	}

	cudaMemcpy(cu_CVVoxels,	ho_CVVoxels,	s_CVVoxels,	cudaMemcpyHostToDevice); cudaCheckError();//Time Consuming!!
}

__host__ void  STomoNV_CUDA::Step1_RotateCVVoxels(SlotDataIterator sdIt, int yprID_to_start)
{
	if(yprID_to_start >= nYPR) return;

	int threads = 256;
	int blocks = min( (int)(nCVVoxel + threads - 1) / threads, 2048);
	const dim3 dgStep1(	blocks);
	const dim3 dbStep1(threads);

	sdIt->SetZero();
	cu_rotCVVoxel_Streamed_16x16 << < dgStep1, dbStep1, 0, sdIt->stream >> > (
					nVoxelX, nCVVoxel, nYPR, //constants
					yprID_to_start,			
					cu_m4x3, cu_CVVoxels,//input data
					sdIt->cu_sNPxl, //output 
					sdIt->cu_sdType,
					sdIt->cu_sdZcrd,
					sdIt->cu_sdZnrm	); cudaCheckError();
}

//----------------------------------------------------------------------------------------------
//
//	bed structure
// 
//----------------------------------------------------------------------------------------------

__host__ void  STomoNV_CUDA::Step3_GenerateBed(SlotDataIterator sdIt)
{
	if(printer_info.BedType == enumBedType::ebtNone) return;

	int threads = 256;
	int blocks = min( (int)(nSlot + threads - 1) / threads, 2048);
	const dim3 dgStep3(	blocks);
	const dim3 dbStep3(threads);

	cu_genBed<< < dgStep3, dbStep3, 0, sdIt->stream >> > (
							nVoxelX, nSlot, //constants
							printer_info.BedType, printer_info.BedOuterBound, printer_info.BedInnerBound, printer_info.BedThickness,//bed parameter
							sdIt->cu_sNPxl,			//slot data
							sdIt->cu_sdType,
							sdIt->cu_sdZcrd,
							sdIt->cu_sdZnrm,
							sdIt->cu_sVo,
							sdIt->cu_sVss); cudaCheckError();
}
